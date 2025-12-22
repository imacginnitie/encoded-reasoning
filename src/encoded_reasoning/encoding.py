"""Unified interface for getting encodings (programmatic ciphers or prompt encodings)."""

from typing import Any, Union

from encoded_reasoning.ciphers import Cipher, get_cipher
from encoded_reasoning.prompt_encodings import PromptEncoding, get_prompt_encoding


def get_encoding(encoding_type: str, **kwargs) -> Union[Cipher, PromptEncoding]:
    """
    Get an encoding instance (programmatic cipher or prompt encoding) by type.

    This function tries to get a programmatic cipher first, then falls back to
    prompt encodings if not found.

    Args:
        encoding_type: Type of encoding (e.g., "caesar", "emojispeak")
        **kwargs: Additional arguments for the encoding (e.g., shift for Caesar)

    Returns:
        Either a Cipher or PromptEncoding instance

    Raises:
        ValueError: If encoding type is not found in either category
    """
    encoding_type_lower = encoding_type.lower()

    # Try programmatic ciphers first
    try:
        return get_cipher(encoding_type_lower, **kwargs)
    except ValueError:
        pass

    # Try prompt encodings
    try:
        return get_prompt_encoding(encoding_type_lower)
    except ValueError:
        pass

    # If neither worked, raise error
    raise ValueError(
        f"Unknown encoding type: {encoding_type}. "
        "Must be either a programmatic cipher or prompt encoding."
    )


def is_programmatic(encoding: Union[Cipher, PromptEncoding]) -> bool:
    """Check if an encoding is programmatic (can encode/decode text)."""
    if hasattr(encoding, "is_programmatic"):
        return encoding.is_programmatic()
    # Default: if it has encode/decode methods, assume it's programmatic
    return hasattr(encoding, "encode") and hasattr(encoding, "decode")

