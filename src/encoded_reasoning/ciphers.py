"""Cipher encoding and decoding functions."""

from typing import Protocol


class Cipher(Protocol):
    """Protocol for cipher implementations."""

    def encode(self, text: str) -> str:
        """Encode text using the cipher."""
        ...

    def decode(self, text: str) -> str:
        """Decode text using the cipher."""
        ...
    def is_programmatic(self) -> bool:
        """Return True - this is a programmatic cipher."""
        return True


class CaesarCipher:
    """Caesar cipher implementation."""

    def __init__(self, shift: int = 3):
        """Initialize Caesar cipher with shift amount."""
        self.shift = shift

    def encode(self, text: str) -> str:
        """Encode text using Caesar cipher."""
        result = []
        for char in text:
            if char.isalpha():
                ascii_offset = 65 if char.isupper() else 97
                shifted = (ord(char) - ascii_offset + self.shift) % 26
                result.append(chr(shifted + ascii_offset))
            else:
                result.append(char)
        return "".join(result)

    def decode(self, text: str) -> str:
        """Decode text using Caesar cipher."""
        result = []
        for char in text:
            if char.isalpha():
                ascii_offset = 65 if char.isupper() else 97
                shifted = (ord(char) - ascii_offset - self.shift) % 26
                result.append(chr(shifted + ascii_offset))
            else:
                result.append(char)
        return "".join(result)

    def get_instruction(self) -> str:
        """Get instruction prompt for this cipher."""
        return f"Use the Caesar cipher with shift {self.shift} to encode/decode text."

    def is_programmatic(self) -> bool:
        """Return True - this is a programmatic cipher."""
        return True


def get_cipher(cipher_type: str, **kwargs) -> Cipher:
    """Get a programmatic cipher instance by type."""
    cipher_type_lower = cipher_type.lower()
    if cipher_type_lower == "caesar":
        shift = kwargs.get("shift", 3)
        return CaesarCipher(shift=shift)
    else:
        raise ValueError(f"Unknown cipher type: {cipher_type}")
