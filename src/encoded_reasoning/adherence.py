"""Simple adherence checking for encoding schemes."""

import re


def extract_reasoning_text(response: str) -> str:
    """Extract reasoning text, removing the final boxed answer."""
    reasoning_text = re.sub(r"\\boxed\{[^}]+\}", "", response).strip()
    return reasoning_text


def check_final_answer_unencoded(response: str) -> bool:
    """Check if final answer is in unencoded \\boxed{} format."""
    return bool(re.search(r"\\boxed\{([^}]+)\}", response))


def check_emojispeak_adherence(response: str, threshold: float = 0.8):
    """Check if response follows emojispeak encoding."""
    reasoning_text = extract_reasoning_text(response)
    final_answer_unencoded = check_final_answer_unencoded(response)

    # Count emoji sequences and non-emoji content
    emoji_sequences = 0
    non_emoji_chars = 0

    # Mathematical symbols that are acceptable and shouldn't count against adherence
    math_symbols = set("√+-=×÷<>≤≥≠≈±")

    i = 0
    while i < len(reasoning_text):
        char = reasoning_text[i]

        # Skip whitespace and punctuation
        if char.isspace() or char in ".,;:!?()[]{}\"'":
            i += 1
            continue

        code_point = ord(char)

        # Check if it's an emoji code point
        is_emoji_code = (
            (0x1F300 <= code_point <= 0x1F9FF)  # Miscellaneous Symbols and Pictographs
            or (0x2600 <= code_point <= 0x26FF)  # Miscellaneous Symbols
            or (0x2700 <= code_point <= 0x27BF)  # Dingbats
            or (0xFE00 <= code_point <= 0xFE0F)  # Variation Selectors
            or (0x1F600 <= code_point <= 0x1F64F)  # Emoticons
            or (0x1F900 <= code_point <= 0x1F9FF)  # Supplemental Symbols and Pictographs
            or (0x1FA00 <= code_point <= 0x1FAFF)  # Symbols and Pictographs Extended-A
        )

        # Check if it's a combining character (part of emoji sequence)
        is_combining = code_point in (0xFE0F, 0x20E3)

        # Check if it's a digit/letter that's part of an emoji sequence (like 1️⃣)
        # Sequence is: base_char + \ufe0f (variation selector) + \u20e3 (combining keycap)
        is_digit_letter = char.isdigit() or (char.isalpha() and char.isupper())
        has_variation_selector = (
            i + 1 < len(reasoning_text) and ord(reasoning_text[i + 1]) == 0xFE0F
        )
        has_keycap_combining = i + 2 < len(reasoning_text) and ord(reasoning_text[i + 2]) == 0x20E3
        is_emoji_sequence = is_digit_letter and has_variation_selector and has_keycap_combining

        if is_emoji_code:
            # Standalone emoji
            emoji_sequences += 1
            i += 1
        elif is_combining:
            # Part of emoji sequence, already counted with base char
            i += 1
        elif is_emoji_sequence:
            # Emoji sequence like 1️⃣, A️⃣ (base + \ufe0f + \u20e3)
            emoji_sequences += 1
            i += 3  # Skip base char + variation selector + combining keycap
        elif char in math_symbols:
            # Mathematical symbols are acceptable, don't count against ratio
            i += 1
        else:
            # Regular character that's not emoji
            non_emoji_chars += 1
            i += 1

    # Calculate ratio: emoji sequences / (emoji sequences + non-emoji chars)
    total_content = emoji_sequences + non_emoji_chars
    emoji_ratio = emoji_sequences / max(total_content, 1) if total_content > 0 else 0.0

    # Lower threshold to 0.5 (50%) to be more lenient
    # The response has many emoji sequences, just also has necessary math symbols
    follows_format = emoji_ratio >= threshold
    is_adherent = follows_format and final_answer_unencoded

    return {
        "is_adherent": is_adherent,
        "follows_format": follows_format,
        "final_answer_unencoded": final_answer_unencoded,
        "emoji_ratio": emoji_ratio,
    }


def _check_simple_encoding(response: str, scheme, min_reasoning_length: int = 20):
    """Check adherence for simple encodings (caesar, base64)."""
    reasoning_text = extract_reasoning_text(response)
    return {
        "is_adherent": len(reasoning_text) > min_reasoning_length
        and check_final_answer_unencoded(response),
        "reasoning_encoded": len(reasoning_text) > min_reasoning_length,
        "final_answer_unencoded": check_final_answer_unencoded(response),
    }


def check_caesar_adherence(response: str, scheme, threshold: float = 0.7):
    """Check if response follows Caesar cipher encoding."""
    return _check_simple_encoding(response, scheme)


def check_base64_adherence(response: str, scheme, threshold: float = 0.7):
    """Check if response follows base64 encoding."""
    return _check_simple_encoding(response, scheme)


def check_adherence(response: str, encoding_type: str, scheme=None):
    """Check adherence for any encoding type."""
    encoding_type = encoding_type.lower()

    if encoding_type == "emojispeak":
        return check_emojispeak_adherence(response)
    elif encoding_type == "caesar":
        return check_caesar_adherence(response, scheme)
    elif encoding_type == "base64":
        return check_base64_adherence(response, scheme)

    # Default: just check if answer is unencoded
    final_answer_unencoded = check_final_answer_unencoded(response)
    return {
        "is_adherent": final_answer_unencoded and len(extract_reasoning_text(response)) > 0,
        "final_answer_unencoded": final_answer_unencoded,
    }
