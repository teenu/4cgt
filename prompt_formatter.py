"""
NoobAI XL V-Pred 1.0 - Prompt Formatter

This module contains the IndexedPromptFormatterData class for loading and
searching character and artist data from CSV files.
"""

import csv
import random
import threading
from typing import List, Dict, Optional, Any
from config import logger, PANDAS_AVAILABLE, SEARCH_CONFIG, SearchScoring
from state import perf_monitor
from utils import CSV_PATHS


def _is_valid_core_tags(value: Any) -> bool:
    """Check if core_tags value is valid (not None, empty, or NaN)."""
    if not value:
        return False
    str_value = str(value).lower()
    return str_value != 'nan' and str_value.strip() != ''

# Check pandas availability
if PANDAS_AVAILABLE:
    import pandas as pd

# ============================================================================
# INDEXED PROMPT FORMATTER
# ============================================================================

class IndexedPromptFormatterData:
    """Prompt formatter with search indexing."""

    def __init__(self):
        self.character_data = {'danbooru': [], 'e621': []}
        self.artist_data = {'danbooru': [], 'e621': []}
        self.char_index = {'danbooru': {}, 'e621': {}}
        self.artist_index = {'danbooru': {}, 'e621': {}}
        self.is_loaded = False
        self.load_data()

    def load_data(self):
        """Load CSV data."""
        if not CSV_PATHS:
            return

        try:
            with perf_monitor.time_section("csv_loading"):
                if PANDAS_AVAILABLE:
                    self._load_with_pandas()
                else:
                    self._load_with_csv()

                self._build_indices()
                self.is_loaded = True

        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            self.is_loaded = False

    def _load_with_pandas(self):
        """Load CSV data using pandas."""
        for source in ['danbooru', 'e621']:
            # Character data
            char_path = CSV_PATHS.get(f'{source}_character')
            if char_path:
                df = pd.read_csv(char_path)
                for row in df.to_dict('records'):
                    character_entry = {
                        'trigger': str(row.get('trigger', '')),
                        'source': source,
                        'character': str(row.get('character', '')),
                        'copyright': str(row.get('copyright', '')),
                        'core_tags': str(row.get('core_tags', '')) if source == 'danbooru' else ''
                    }
                    self.character_data[source].append(character_entry)

            # Artist data
            artist_path = CSV_PATHS.get(f'{source}_artist')
            if artist_path:
                df = pd.read_csv(artist_path)
                for row in df.to_dict('records'):
                    artist_entry = {
                        'trigger': str(row.get('trigger', '')),
                        'source': source,
                        'artist': str(row.get('artist', ''))
                    }
                    self.artist_data[source].append(artist_entry)

    def _load_with_csv(self):
        """Load CSV data without pandas with robust encoding handling."""
        for source in ['danbooru', 'e621']:
            # Character data
            char_path = CSV_PATHS.get(f'{source}_character')
            if char_path:
                try:
                    # Try UTF-8 first (most common)
                    with open(char_path, 'r', encoding='utf-8', errors='replace') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            character_entry = {
                                'trigger': str(row.get('trigger', '')),
                                'source': source,
                                'character': str(row.get('character', '')),
                                'copyright': str(row.get('copyright', '')),
                                'core_tags': str(row.get('core_tags', '')) if source == 'danbooru' else ''
                            }
                            self.character_data[source].append(character_entry)
                except (UnicodeDecodeError, UnicodeError) as e:
                    logger.warning(f"UTF-8 decode failed for {char_path}, characters may be corrupted: {e}")
                except (IOError, OSError) as e:
                    logger.error(f"Failed to read character CSV {char_path}: {e}")

            # Artist data
            artist_path = CSV_PATHS.get(f'{source}_artist')
            if artist_path:
                try:
                    # Try UTF-8 first (most common)
                    with open(artist_path, 'r', encoding='utf-8', errors='replace') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            artist_entry = {
                                'trigger': str(row.get('trigger', '')),
                                'source': source,
                                'artist': str(row.get('artist', ''))
                            }
                            self.artist_data[source].append(artist_entry)
                except (UnicodeDecodeError, UnicodeError) as e:
                    logger.warning(f"UTF-8 decode failed for {artist_path}, characters may be corrupted: {e}")
                except (IOError, OSError) as e:
                    logger.error(f"Failed to read artist CSV {artist_path}: {e}")

    def _build_indices(self):
        """Build search indices for faster lookups."""
        with perf_monitor.time_section("index_building"):
            # Build character indices
            for source in ['danbooru', 'e621']:
                for item in self.character_data[source]:
                    trigger_lower = item['trigger'].lower()
                    # Index by prefixes for faster prefix matching
                    for i in range(SEARCH_CONFIG.MIN_QUERY_LENGTH,
                                   min(SEARCH_CONFIG.INDEX_PREFIX_LENGTH + 1, len(trigger_lower) + 1)):
                        prefix = trigger_lower[:i]
                        if prefix not in self.char_index[source]:
                            self.char_index[source][prefix] = []
                        # Limit index size to prevent excessive memory usage
                        if len(self.char_index[source][prefix]) < 1000:
                            self.char_index[source][prefix].append(item)

                # Build artist indices
                for item in self.artist_data[source]:
                    trigger_lower = item['trigger'].lower()
                    for i in range(SEARCH_CONFIG.MIN_QUERY_LENGTH,
                                   min(SEARCH_CONFIG.INDEX_PREFIX_LENGTH + 1, len(trigger_lower) + 1)):
                        prefix = trigger_lower[:i]
                        if prefix not in self.artist_index[source]:
                            self.artist_index[source][prefix] = []
                        # Limit index size to prevent excessive memory usage
                        if len(self.artist_index[source][prefix]) < 1000:
                            self.artist_index[source][prefix].append(item)

    def _calculate_search_score(self, query_lower: str, trigger_lower: str) -> int:
        """Calculate search relevance score."""
        if trigger_lower == query_lower:
            return SearchScoring.EXACT_MATCH
        elif trigger_lower.startswith(query_lower):
            return SearchScoring.PREFIX_MATCH
        else:
            return SearchScoring.CONTAINS_MATCH

    def search(self, query: str, data_type: str, limit: int = 10) -> List[Dict]:
        """Search for entries with optimized indexing."""
        if not self.is_loaded or not query.strip():
            return []

        with perf_monitor.time_section(f"search_{data_type}"):
            query = query.strip()[:SEARCH_CONFIG.MAX_QUERY_LENGTH]
            limit = min(limit, SEARCH_CONFIG.MAX_RESULTS)

            if len(query) < SEARCH_CONFIG.MIN_QUERY_LENGTH:
                return []

            query_lower = query.lower()

            # Select appropriate index and data
            if data_type == 'character':
                index = self.char_index
                full_data = self.character_data
            else:
                index = self.artist_index
                full_data = self.artist_data

            all_results = []
            seen_triggers = set()

            # First, check indexed entries for prefix matches
            for source in ['danbooru', 'e621']:
                # Use index for fast prefix lookup
                prefix_key = query_lower[:min(len(query_lower), SEARCH_CONFIG.INDEX_PREFIX_LENGTH)]
                indexed_items = index[source].get(prefix_key, [])

                for item in indexed_items:
                    trigger_lower = item['trigger'].lower()
                    if trigger_lower in seen_triggers:
                        continue

                    if query_lower in trigger_lower:
                        score = self._calculate_search_score(query_lower, trigger_lower)
                        result = {
                            'display': item['trigger'],
                            'source': source,
                            'score': score
                        }

                        # Add value based on type and source
                        core_tags = item.get('core_tags', '')
                        if data_type == 'character' and source == 'danbooru' and _is_valid_core_tags(core_tags):
                            result['value'] = f"{core_tags}, {item['trigger']}"
                        else:
                            result['value'] = item['trigger']

                        all_results.append(result)
                        seen_triggers.add(trigger_lower)

                # Fallback to full search if not enough results
                if len(all_results) < limit:
                    for item in full_data[source]:
                        # Early termination once we have enough results
                        if len(all_results) >= limit:
                            break

                        trigger_lower = item['trigger'].lower()
                        if trigger_lower in seen_triggers:
                            continue

                        if query_lower in trigger_lower:
                            score = self._calculate_search_score(query_lower, trigger_lower)
                            result = {
                                'display': item['trigger'],
                                'source': source,
                                'score': score
                            }

                            core_tags = item.get('core_tags', '')
                            if data_type == 'character' and source == 'danbooru' and _is_valid_core_tags(core_tags):
                                result['value'] = f"{core_tags}, {item['trigger']}"
                            else:
                                result['value'] = item['trigger']

                            all_results.append(result)
                            seen_triggers.add(trigger_lower)

            # Sort by score and source
            all_results.sort(key=lambda x: (x['score'], x['source']), reverse=True)

            # Balance results between sources
            final_results = []
            danbooru_count = 0
            e621_count = 0
            max_per_source = (limit + 1) // 2

            for result in all_results:
                if result['source'] == 'danbooru' and danbooru_count < max_per_source:
                    final_results.append(result)
                    danbooru_count += 1
                elif result['source'] == 'e621' and e621_count < max_per_source:
                    final_results.append(result)
                    e621_count += 1

                if len(final_results) >= limit:
                    break

            # Remove score from final results
            return [{k: v for k, v in r.items() if k != 'score'} for r in final_results]

    def get_random_entry(self, data_type: str, source_filter: str = None) -> Optional[Dict]:
        """
        Get a random entry with 50/50 source balancing.

        Args:
            data_type: 'character' or 'artist'
            source_filter: None (balanced), 'danbooru', or 'e621'

        Returns:
            Dict with 'display', 'source', 'value' keys, or None if no data
        """
        if not self.is_loaded:
            return None

        # Select data based on type
        if data_type == 'character':
            data = self.character_data
        elif data_type == 'artist':
            data = self.artist_data
        else:
            return None

        # Determine which source to use
        if source_filter in ['danbooru', 'e621']:
            source = source_filter
        else:
            # 50/50 balanced selection - randomly pick source first
            available_sources = [s for s in ['danbooru', 'e621'] if data[s]]
            if not available_sources:
                return None
            source = random.choice(available_sources)

        entries = data[source]

        if not entries:
            # Fallback to other source if selected is empty
            other_source = 'e621' if source == 'danbooru' else 'danbooru'
            entries = data[other_source]
            source = other_source

        if not entries:
            return None

        # Select random entry
        item = random.choice(entries)

        # Format the value (same logic as search results)
        core_tags = item.get('core_tags', '')
        if data_type == 'character' and source == 'danbooru' and _is_valid_core_tags(core_tags):
            value = f"{core_tags}, {item['trigger']}"
        else:
            value = item['trigger']

        return {
            'display': item['trigger'],
            'source': source,
            'value': value
        }

# Create global instance with thread-safe lazy initialization
prompt_formatter_data: Optional[IndexedPromptFormatterData] = None
_prompt_data_lock = threading.Lock()

def get_prompt_data() -> IndexedPromptFormatterData:
    """Get or create prompt formatter data instance (thread-safe)."""
    global prompt_formatter_data

    # Double-checked locking pattern for thread-safe lazy initialization
    if prompt_formatter_data is None:
        with _prompt_data_lock:
            # Check again inside lock to prevent race condition
            if prompt_formatter_data is None:
                prompt_formatter_data = IndexedPromptFormatterData()

    return prompt_formatter_data
