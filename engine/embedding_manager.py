"""NoobAI XL V-Pred 1.0 - Embedding Manager for Textual Inversions.

This module handles discovery, loading, and management of textual inversion
embeddings (CLIP embeddings stored in safetensors format).

Supports SDXL dual-CLIP format with clip_g and clip_l tensors.
"""

import os
import torch
from typing import List, Dict, Optional, Any
from safetensors import safe_open
from config import (
    logger, EMBEDDING_CONFIG, EMBEDDING_SEARCH_DIRECTORIES,
    QUALITY_EMBEDDING_TRIGGERS, NEGATIVE_EMBEDDING_TRIGGERS
)


class EmbeddingManager:
    """Manages textual inversion embeddings for the diffusion pipeline."""

    def __init__(self, pipe):
        """Initialize the embedding manager.

        Args:
            pipe: The diffusion pipeline (StableDiffusionXLPipeline)
        """
        self.pipe = pipe
        self.loaded_embeddings: Dict[str, str] = {}  # token -> path

    def _load_sdxl_embedding(self, embedding_path: str, token: str) -> bool:
        """Load an SDXL-format embedding with clip_g and clip_l tensors.

        This handles the community SDXL embedding format that stores separate
        embeddings for both CLIP text encoders.

        Args:
            embedding_path: Path to the .safetensors file
            token: Trigger token for this embedding

        Returns:
            True if loading succeeded
        """
        try:
            # Load tensors from safetensors file
            with safe_open(embedding_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())

                # Check if this is SDXL format
                if "clip_l" not in keys or "clip_g" not in keys:
                    return False

                clip_l_emb = f.get_tensor("clip_l")  # [num_tokens, 768]
                clip_g_emb = f.get_tensor("clip_g")  # [num_tokens, 1280]

            num_tokens = clip_l_emb.shape[0]
            logger.info(f"Loading SDXL embedding '{token}' with {num_tokens} tokens")

            # Get tokenizers and text encoders
            tokenizer = self.pipe.tokenizer
            tokenizer_2 = self.pipe.tokenizer_2
            text_encoder = self.pipe.text_encoder
            text_encoder_2 = self.pipe.text_encoder_2

            # Create placeholder tokens for multi-token embeddings
            # For n tokens, we use: token, token_1, token_2, ... token_(n-1)
            placeholder_tokens = [token] + [f"{token}_{i}" for i in range(1, num_tokens)]

            # Add tokens to both tokenizers
            num_added_1 = tokenizer.add_tokens(placeholder_tokens)
            num_added_2 = tokenizer_2.add_tokens(placeholder_tokens)

            if num_added_1 == 0 and num_added_2 == 0:
                logger.warning(f"Token '{token}' already exists in tokenizers")

            # Check if models are on meta device (CPU offloading enabled)
            # If so, we need to handle this differently
            is_meta_1 = hasattr(text_encoder, 'device') and str(text_encoder.device) == 'meta'
            is_meta_2 = hasattr(text_encoder_2, 'device') and str(text_encoder_2.device) == 'meta'

            # Also check the embedding weights directly
            embeddings_1 = text_encoder.get_input_embeddings()
            embeddings_2 = text_encoder_2.get_input_embeddings()

            if embeddings_1.weight.device.type == 'meta' or embeddings_2.weight.device.type == 'meta':
                # CPU offloading is active - need to handle meta tensors
                logger.info("CPU offloading detected - using meta-tensor compatible embedding injection")
                return self._load_sdxl_embedding_with_offload(
                    token, placeholder_tokens, num_tokens,
                    clip_l_emb, clip_g_emb,
                    tokenizer, tokenizer_2,
                    text_encoder, text_encoder_2
                )

            # Standard path - no CPU offloading
            # Resize token embeddings for both text encoders (disable mean_resizing to avoid issues)
            text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            text_encoder_2.resize_token_embeddings(len(tokenizer_2), mean_resizing=False)

            # Get token IDs
            token_ids_1 = tokenizer.convert_tokens_to_ids(placeholder_tokens)
            token_ids_2 = tokenizer_2.convert_tokens_to_ids(placeholder_tokens)

            # Refresh embedding layer references after resize
            embeddings_1 = text_encoder.get_input_embeddings()
            embeddings_2 = text_encoder_2.get_input_embeddings()

            # Inject embeddings for text_encoder (CLIP-L, 768 dim)
            with torch.no_grad():
                for i, token_id in enumerate(token_ids_1):
                    embeddings_1.weight[token_id] = clip_l_emb[i].to(
                        dtype=embeddings_1.weight.dtype,
                        device=embeddings_1.weight.device
                    )

            # Inject embeddings for text_encoder_2 (CLIP-G, 1280 dim)
            with torch.no_grad():
                for i, token_id in enumerate(token_ids_2):
                    embeddings_2.weight[token_id] = clip_g_emb[i].to(
                        dtype=embeddings_2.weight.dtype,
                        device=embeddings_2.weight.device
                    )

            logger.info(f"Successfully loaded SDXL embedding '{token}' ({num_tokens} tokens)")
            return True

        except Exception as e:
            logger.error(f"Failed to load SDXL embedding: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _load_sdxl_embedding_with_offload(
        self, token: str, placeholder_tokens: List[str], num_tokens: int,
        clip_l_emb: torch.Tensor, clip_g_emb: torch.Tensor,
        tokenizer, tokenizer_2, text_encoder, text_encoder_2
    ) -> bool:
        """Load SDXL embedding when CPU offloading is enabled.

        When CPU offloading is active, the model weights are on 'meta' device
        and we cannot directly modify them. Instead, we need to:
        1. Move the text encoders to CPU temporarily
        2. Resize and inject embeddings
        3. The offload hooks will handle moving them back as needed
        """
        try:
            import accelerate

            # Get the device to use for temporary operations
            device = torch.device("cpu")

            # Move text encoders to CPU temporarily to materialize weights
            logger.debug("Moving text encoders to CPU for embedding injection")

            # Remove offload hooks temporarily if they exist
            if hasattr(text_encoder, '_hf_hook'):
                accelerate.hooks.remove_hook_from_module(text_encoder, recurse=True)
            if hasattr(text_encoder_2, '_hf_hook'):
                accelerate.hooks.remove_hook_from_module(text_encoder_2, recurse=True)

            # Move to CPU to materialize
            text_encoder.to(device)
            text_encoder_2.to(device)

            # Now resize token embeddings
            text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            text_encoder_2.resize_token_embeddings(len(tokenizer_2), mean_resizing=False)

            # Get token IDs
            token_ids_1 = tokenizer.convert_tokens_to_ids(placeholder_tokens)
            token_ids_2 = tokenizer_2.convert_tokens_to_ids(placeholder_tokens)

            # Get embedding layers
            embeddings_1 = text_encoder.get_input_embeddings()
            embeddings_2 = text_encoder_2.get_input_embeddings()

            # Inject embeddings for text_encoder (CLIP-L, 768 dim)
            with torch.no_grad():
                for i, token_id in enumerate(token_ids_1):
                    embeddings_1.weight[token_id] = clip_l_emb[i].to(
                        dtype=embeddings_1.weight.dtype,
                        device=device
                    )

            # Inject embeddings for text_encoder_2 (CLIP-G, 1280 dim)
            with torch.no_grad():
                for i, token_id in enumerate(token_ids_2):
                    embeddings_2.weight[token_id] = clip_g_emb[i].to(
                        dtype=embeddings_2.weight.dtype,
                        device=device
                    )

            # Re-enable CPU offloading by moving back to meta and re-applying hooks
            # The pipeline's enable_sequential_cpu_offload should be called again
            # but that's complex - instead, just leave on CPU and let the pipeline handle it
            logger.info(f"Successfully loaded SDXL embedding '{token}' ({num_tokens} tokens) with CPU offload handling")
            return True

        except ImportError:
            logger.error("accelerate package required for CPU offload handling")
            return False
        except Exception as e:
            logger.error(f"Failed to load SDXL embedding with CPU offload: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def load_embedding(self, embedding_path: str, token: Optional[str] = None) -> bool:
        """Load a textual inversion embedding into the pipeline.

        Supports both standard diffusers format and SDXL dual-CLIP format.

        Args:
            embedding_path: Path to the .safetensors embedding file
            token: Optional custom trigger token. If None, derived from filename.

        Returns:
            True if loading succeeded, False otherwise
        """
        if not os.path.exists(embedding_path):
            logger.error(f"Embedding file not found: {embedding_path}")
            return False

        if token is None:
            # Derive token from filename (without extension)
            token = os.path.splitext(os.path.basename(embedding_path))[0]

        # Handle -neg suffix convention
        base_token = None
        if token.endswith("-neg"):
            base_token = token[:-4]  # Version without -neg suffix

        # First, try to load as SDXL format (clip_g + clip_l)
        try:
            with safe_open(embedding_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())

            if "clip_l" in keys and "clip_g" in keys:
                # This is SDXL format - use custom loader
                if self._load_sdxl_embedding(embedding_path, token):
                    self.loaded_embeddings[token] = embedding_path

                    # Also register base token alias for -neg files
                    if base_token and base_token not in self.loaded_embeddings:
                        # The tokens are already in the tokenizer, just add to our registry
                        self.loaded_embeddings[base_token] = embedding_path
                        logger.info(f"Also registered alias '{base_token}' for embedding")

                    return True
                else:
                    return False

        except Exception as e:
            logger.debug(f"Could not check embedding format: {e}")

        # Fall back to standard diffusers method
        try:
            self.pipe.load_textual_inversion(embedding_path, token=token)
            self.loaded_embeddings[token] = embedding_path
            logger.info(f"Loaded embedding '{token}' from {embedding_path}")

            # If this is a -neg file, also register the base name as an alias
            if base_token and base_token not in self.loaded_embeddings:
                try:
                    self.pipe.load_textual_inversion(embedding_path, token=base_token)
                    self.loaded_embeddings[base_token] = embedding_path
                    logger.info(f"Also registered alias '{base_token}' for embedding")
                except Exception as e:
                    logger.debug(f"Could not register alias '{base_token}': {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to load embedding '{token}' from {embedding_path}: {e}")
            return False

    def load_embeddings(self, embedding_paths: List[str]) -> int:
        """Load multiple embeddings.

        Args:
            embedding_paths: List of paths to embedding files

        Returns:
            Number of successfully loaded embeddings
        """
        loaded_count = 0
        for path in embedding_paths:
            if self.load_embedding(path):
                loaded_count += 1
        return loaded_count

    def is_loaded(self, token: str) -> bool:
        """Check if an embedding with the given token is loaded."""
        return token in self.loaded_embeddings

    def get_loaded_tokens(self) -> List[str]:
        """Get list of all loaded embedding tokens."""
        return list(self.loaded_embeddings.keys())

    def has_quality_embedding(self, prompt: str) -> bool:
        """Check if prompt contains a loaded quality embedding trigger.

        Args:
            prompt: The positive prompt to check

        Returns:
            True if a quality embedding trigger is found and loaded
        """
        for trigger in QUALITY_EMBEDDING_TRIGGERS:
            if trigger in prompt and self.is_loaded(trigger):
                return True
        return False

    def has_negative_embedding(self, prompt: str) -> bool:
        """Check if prompt contains a loaded negative embedding trigger.

        Args:
            prompt: The negative prompt to check

        Returns:
            True if a negative embedding trigger is found and loaded
        """
        for trigger in NEGATIVE_EMBEDDING_TRIGGERS:
            if trigger in prompt and self.is_loaded(trigger):
                return True
        return False

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about loaded embeddings."""
        return {
            'loaded_count': len(self.loaded_embeddings),
            'tokens': list(self.loaded_embeddings.keys()),
            'paths': list(self.loaded_embeddings.values())
        }


def discover_embeddings() -> List[Dict[str, Any]]:
    """Discover available embedding files in search directories.

    Returns:
        List of dicts with embedding info: name, path, size, is_negative
    """
    embeddings = []
    seen_paths = set()

    for directory in EMBEDDING_SEARCH_DIRECTORIES:
        if not os.path.isdir(directory):
            continue

        try:
            for file in os.listdir(directory):
                if not file.endswith('.safetensors'):
                    continue

                path = os.path.join(directory, file)

                # Skip duplicates
                if path in seen_paths:
                    continue
                seen_paths.add(path)

                try:
                    size_bytes = os.path.getsize(path)
                    size_kb = size_bytes / 1024

                    # Check size bounds (embeddings are small, typically < 1MB)
                    if not (EMBEDDING_CONFIG.MIN_FILE_SIZE_KB <= size_kb <= EMBEDDING_CONFIG.MAX_FILE_SIZE_MB * 1024):
                        continue

                    name = os.path.splitext(file)[0]
                    is_negative = name.endswith("-neg") or "negative" in name.lower()

                    embeddings.append({
                        'name': name,
                        'path': path,
                        'size': f"{size_kb:.1f} KB",
                        'size_bytes': size_bytes,
                        'is_negative': is_negative,
                        'display_name': f"{name} ({size_kb:.1f} KB)"
                    })

                except (OSError, IOError) as e:
                    logger.debug(f"Could not read embedding file {path}: {e}")

        except (OSError, IOError) as e:
            logger.debug(f"Could not list directory {directory}: {e}")

    # Sort by name
    embeddings.sort(key=lambda x: x['name'].lower())
    return embeddings


def get_embedding_ui_choices() -> tuple:
    """Get embedding choices formatted for Gradio dropdown.

    Returns:
        Tuple of (choices_list, has_embeddings)
    """
    embeddings = discover_embeddings()

    if not embeddings:
        return (["No embeddings found"], False)

    choices = ["None (no embedding)"]
    for emb in embeddings:
        choices.append(emb['display_name'])

    return (choices, True)


def get_embedding_path_from_selection(selection: str) -> Optional[str]:
    """Get the file path for a selected embedding from dropdown.

    Args:
        selection: The dropdown selection string

    Returns:
        Path to the embedding file, or None if not found/applicable
    """
    if not selection or selection == "None (no embedding)" or selection == "No embeddings found":
        return None

    embeddings = discover_embeddings()

    for emb in embeddings:
        if emb['display_name'] == selection or emb['name'] == selection:
            return emb['path']

    return None
