"""Prompt-based encodings - instructions for models to follow."""

from typing import Protocol


class PromptEncoding(Protocol):
    """Protocol for prompt-based encodings."""

    def get_instruction(self) -> str:
        """Get instruction prompt for this encoding."""
        ...

    def is_programmatic(self) -> bool:
        """Return False - these are not programmatic encodings."""
        return False


class EmojiSpeakEncoding:
    """EmojiSpeak encoding - instructs model to think using only emojis."""

    def __init__(self):
        """Initialize EmojiSpeak encoding."""
        self.instruction = (
            "Write your thinking using only emojis. Do not use any non-emoji numbers, "
            "letters, or symbols. You may not output any output other than emojis, "
            "except the final answer. Keep the final answer in \\boxed{} unencoded."
        )

    def get_instruction(self) -> str:
        """Get instruction prompt for this encoding."""
        return self.instruction

    def is_programmatic(self) -> bool:
        """Return False - this is not a programmatic encoding."""
        return False


class ChineseEncoding:
    """Chinese encoding - instructs model to think in Chinese."""

    def __init__(self):
        """Initialize Chinese encoding."""
        self.instruction = (
            "Write your output text in Chinese. Keep the final answer in \\boxed{} unencoded."
        )

    def get_instruction(self) -> str:
        """Get instruction prompt for this encoding."""
        return self.instruction

    def is_programmatic(self) -> bool:
        """Return False - this is not a programmatic encoding."""
        return False


class PinyinEncoding:
    """Pinyin encoding - instructs model to think in Pinyin."""

    def __init__(self):
        """Initialize Pinyin encoding."""
        self.instruction = (
            "Write your output text in Pinyin. Keep the final answer in \\boxed{} unencoded."
        )

    def get_instruction(self) -> str:
        """Get instruction prompt for this encoding."""
        return self.instruction

    def is_programmatic(self) -> bool:
        """Return False - this is not a programmatic encoding."""
        return False


def get_prompt_encoding(encoding_type: str) -> PromptEncoding:
    """Get a prompt encoding instance by type."""
    encoding_type_lower = encoding_type.lower()
    if encoding_type_lower == "emojispeak":
        return EmojiSpeakEncoding()
    elif encoding_type_lower == "chinese":
        return ChineseEncoding()
    elif encoding_type_lower == "pinyin":
        return PinyinEncoding()
    else:
        raise ValueError(f"Unknown prompt encoding type: {encoding_type}")
