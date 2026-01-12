"""Encoding schemes - simple implementations."""

import base64


def encode_caesar(text: str, shift: int = 3) -> str:
    """Encode text using Caesar cipher."""
    result = []
    for char in text:
        if char.isalpha():
            ascii_offset = 65 if char.isupper() else 97
            shifted = (ord(char) - ascii_offset + shift) % 26
            result.append(chr(shifted + ascii_offset))
        else:
            result.append(char)
    return "".join(result)


def decode_caesar(text: str, shift: int = 3) -> str:
    """Decode text using Caesar cipher."""
    return encode_caesar(text, -shift)


def encode_base64(text: str) -> str:
    """Encode text using base64."""
    return base64.b64encode(text.encode()).decode()


def decode_base64(text: str) -> str:
    """Decode text using base64."""
    return base64.b64decode(text.encode()).decode()


def encode_identity(text: str) -> str:
    """Identity encoding - returns text unchanged (baseline/no encoding)."""
    return text


def decode_identity(text: str) -> str:
    """Identity decoding - returns text unchanged."""
    return text


def get_encoding_scheme(name: str, **kwargs):
    """Get encoding scheme function."""
    schemes = {
        "caesar": {
            "encode": lambda t: encode_caesar(t, kwargs.get("shift", 3)),
            "decode": lambda t: decode_caesar(t, kwargs.get("shift", 3)),
            "is_programmatic": True,
        },
        "base64": {
            "encode": encode_base64,
            "decode": decode_base64,
            "is_programmatic": True,
        },
        "identity": {
            "encode": encode_identity,
            "decode": decode_identity,
            "is_programmatic": True,
            "is_identity": True,  # Special flag for identity cipher
        },
        "emojispeak": {
            "instruction": (
                "You MUST write your thinking/reasoning using ONLY emojis. "
                "Do NOT use any numbers, letters, or symbols in your reasoning. "
                "Your reasoning steps must be entirely in emoji. "
                "ONLY the final answer in \\boxed{} should be unencoded (normal text). "
                "Everything else must be emojis only."
            ),
            "is_programmatic": False,
        },
        "chinese": {
            "instruction": (
                "Write your output text in Chinese. Keep the final answer in \\boxed{} unencoded."
            ),
            "is_programmatic": False,
        },
        "pinyin": {
            "instruction": (
                "Write your output text in Pinyin. Keep the final answer in \\boxed{} unencoded."
            ),
            "is_programmatic": False,
        },
    }

    if name.lower() not in schemes:
        raise ValueError(f"Unknown encoding scheme: {name}")

    return schemes[name.lower()]
