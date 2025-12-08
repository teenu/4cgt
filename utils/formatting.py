"""Text and file formatting utilities."""

import hashlib
import unicodedata
from config import logger


def normalize_text(text: str) -> str:
    """Normalize Unicode text and strip whitespace."""
    if not text:
        return ""
    return unicodedata.normalize('NFC', text.strip())


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable binary format (1 KiB = 1024 bytes)."""
    if not isinstance(size_bytes, (int, float)):
        return "0.00 B"
    if size_bytes < 0:
        return "0.00 B"
    for unit in ['B', 'KiB', 'MiB', 'GiB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TiB"


def calculate_image_hash(file_path: str) -> str:
    """Calculate MD5 hash of an image file."""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except (IOError, OSError) as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        return "ERROR"


def get_user_friendly_error(error: Exception) -> str:
    """Convert technical errors to user-friendly messages."""
    from config import USER_FRIENDLY_ERRORS
    error_str = str(error).lower()
    for key, message in USER_FRIENDLY_ERRORS.items():
        if key.lower() in error_str:
            return message
    return str(error)
