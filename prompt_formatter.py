"""
NoobAI XL V-Pred 1.0 - Prompt Formatter

This module contains the IndexedPromptFormatterData class for loading and
searching character and artist data from CSV files.
"""

import csv
from typing import List, Dict, Optional
from config import logger, PANDAS_AVAILABLE, SEARCH_CONFIG, SearchScoring
from state import perf_monitor
from utils import CSV_PATHS

# Check pandas availability
if PANDAS_AVAILABLE:
    import pandas as pd

# ============================================================================
# INDEXED PROMPT FORMATTER
# ============================================================================

class IndexedPromptFormatterData:
    """Enhanced prompt formatter with search indexing for better performance."""

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
                logger.info("CSV data loaded and indexed successfully.")

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
                for _, row in df.iterrows():
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
                for _, row in df.iterrows():
                    artist_entry = {
                        'trigger': str(row.get('trigger', '')),
                        'source': source,
                        'artist': str(row.get('artist', ''))
                    }
                    self.artist_data[source].append(artist_entry)

    def _load_with_csv(self):
        """Load CSV data without pandas."""
        for source in ['danbooru', 'e621']:
            # Character data
            char_path = CSV_PATHS.get(f'{source}_character')
            if char_path:
                with open(char_path, 'r', encoding='utf-8') as f:
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

            # Artist data
            artist_path = CSV_PATHS.get(f'{source}_artist')
            if artist_path:
                with open(artist_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        artist_entry = {
                            'trigger': str(row.get('trigger', '')),
                            'source': source,
                            'artist': str(row.get('artist', ''))
                        }
                        self.artist_data[source].append(artist_entry)

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
                        self.char_index[source][prefix].append(item)

                # Build artist indices
                for item in self.artist_data[source]:
                    trigger_lower = item['trigger'].lower()
                    for i in range(SEARCH_CONFIG.MIN_QUERY_LENGTH,
                                   min(SEARCH_CONFIG.INDEX_PREFIX_LENGTH + 1, len(trigger_lower) + 1)):
                        prefix = trigger_lower[:i]
                        if prefix not in self.artist_index[source]:
                            self.artist_index[source][prefix] = []
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
                        if data_type == 'character' and source == 'danbooru' and item['core_tags']:
                            result['value'] = f"{item['core_tags']}, {item['trigger']}"
                        else:
                            result['value'] = item['trigger']

                        all_results.append(result)
                        seen_triggers.add(trigger_lower)

                # Fallback to full search if not enough results
                if len(all_results) < limit:
                    for item in full_data[source]:
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

                            if data_type == 'character' and source == 'danbooru' and item['core_tags']:
                                result['value'] = f"{item['core_tags']}, {item['trigger']}"
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

# Create global instance
prompt_formatter_data: Optional[IndexedPromptFormatterData] = None

def get_prompt_data() -> IndexedPromptFormatterData:
    """Get or create prompt formatter data instance."""
    global prompt_formatter_data
    if prompt_formatter_data is None:
        prompt_formatter_data = IndexedPromptFormatterData()
    return prompt_formatter_data
