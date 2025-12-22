"""Embedding generation with long prompt support via sd_embed.

This module provides a wrapper around the sd_embed library for generating
text embeddings that support prompts longer than the 77-token CLIP limit.

The sd_embed library handles:
- Automatic chunking of long prompts into 75-token segments
- A1111-compatible weight syntax: (tag:1.2), [tag], ((emphasis))
- Proper handling of both SDXL text encoders (CLIP-L and OpenCLIP-G)
- Correct generation of pooled embeddings required by SDXL

IMPORTANT: To maintain output parity with older versions, sd_embed is ONLY
activated when either the prompt or negative prompt exceeds 77 tokens.
For prompts within the 77-token limit, the standard SDXL pipeline encoding
is used to ensure identical outputs.
"""

from typing import Tuple, Any, Optional
import torch
from config import logger

# Try to import sd_embed, set availability flag
try:
    from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
    SD_EMBED_AVAILABLE = True
except ImportError:
    SD_EMBED_AVAILABLE = False
    logger.info("sd_embed not installed - long prompt support will use fallback mode")


class EmbeddingGenerator:
    """Generate embeddings for SDXL with conditional long prompt support.

    This class wraps sd_embed's get_weighted_text_embeddings_sdxl function
    to provide:
    - Unlimited prompt length via automatic chunking
    - A1111 weight syntax support: (tag:1.2), [tag], ((emphasis))
    - Graceful fallback to standard encoding if sd_embed fails

    IMPORTANT: To maintain output parity with older versions, sd_embed is
    only used when prompts exceed the 77-token CLIP limit. For short prompts,
    the standard SDXL pipeline encoding is used.

    Usage:
        generator = EmbeddingGenerator(pipe)
        embeds = generator.generate(prompt, negative_prompt, use_long_prompt_mode=True)
        # Pass embeds to pipeline instead of raw prompt strings
    """

    CLIP_TOKEN_LIMIT = 77

    def __init__(self, pipe: Any):
        """Initialize with SDXL pipeline.

        Args:
            pipe: StableDiffusionXLPipeline instance with text encoders
        """
        self.pipe = pipe
        self._sd_embed_available = SD_EMBED_AVAILABLE

        if not self._sd_embed_available:
            logger.info(
                "sd_embed not available - long prompts will be truncated to 77 tokens. "
                "Install with: pip install git+https://github.com/xhinker/sd_embed.git@main"
            )

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        prompt_exceeds_limit: bool = False,
        negative_exceeds_limit: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate embeddings for prompt and negative prompt.

        To maintain output parity with older versions, sd_embed is only used
        when either prompt exceeds 77 tokens. For prompts within the limit,
        standard SDXL pipeline encoding is used for identical outputs.

        Args:
            prompt: The positive prompt text
            negative_prompt: The negative prompt text
            prompt_exceeds_limit: True if prompt has > 77 tokens
            negative_exceeds_limit: True if negative prompt has > 77 tokens

        Returns:
            Tuple of 4 tensors:
            - prompt_embeds: Encoded positive prompt
            - negative_prompt_embeds: Encoded negative prompt
            - pooled_prompt_embeds: Pooled positive embedding
            - negative_pooled_prompt_embeds: Pooled negative embedding
        """
        # Determine if we need sd_embed (either prompt exceeds 77 tokens)
        needs_long_prompt_mode = prompt_exceeds_limit or negative_exceeds_limit

        # Use standard encoding for short prompts (maintains output parity)
        if not needs_long_prompt_mode:
            return self._standard_encode(prompt, negative_prompt)

        # Use sd_embed for long prompts if available
        if not self._sd_embed_available:
            logger.warning(
                "Long prompt detected but sd_embed not available. "
                "Prompt will be truncated to 77 tokens."
            )
            return self._standard_encode(prompt, negative_prompt)

        try:
            # Use sd_embed for long prompt handling
            (prompt_embeds,
             negative_prompt_embeds,
             pooled_prompt_embeds,
             negative_pooled_prompt_embeds) = get_weighted_text_embeddings_sdxl(
                self.pipe,
                prompt=prompt,
                neg_prompt=negative_prompt
            )

            return (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds
            )

        except Exception as e:
            logger.warning(f"sd_embed encoding failed, using standard encoding: {e}")
            return self._standard_encode(prompt, negative_prompt)

    def _standard_encode(
        self,
        prompt: str,
        negative_prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard SDXL pipeline encoding (77-token limit).

        This method uses the pipeline's built-in encode_prompt method.
        Used for prompts within the 77-token limit to maintain output
        parity with older versions that didn't support long prompts.

        Args:
            prompt: The positive prompt text
            negative_prompt: The negative prompt text

        Returns:
            Tuple of 4 tensors (same as generate method)
        """
        try:
            # Determine the device to use
            device = self.pipe.device if hasattr(self.pipe, 'device') else 'cuda'
            if hasattr(device, 'type'):
                device = device

            # Use pipeline's internal encoding method
            # This handles both text encoders but truncates at 77 tokens
            (prompt_embeds,
             negative_prompt_embeds,
             pooled_prompt_embeds,
             negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,  # Same prompt for both encoders
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )

            return (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds
            )

        except Exception as e:
            logger.error(f"Standard encoding failed: {e}")
            raise RuntimeError(f"Prompt encoding failed: {e}")

    @property
    def is_long_prompt_supported(self) -> bool:
        """Check if long prompt support is available.

        Returns:
            True if sd_embed is available for use with long prompts
        """
        return self._sd_embed_available

    @property
    def mode_description(self) -> str:
        """Get a human-readable description of the current mode.

        Returns:
            Description string for logging/display
        """
        if self._sd_embed_available:
            return "standard (sd_embed available for prompts > 77 tokens)"
        else:
            return "standard only (77-token limit)"
