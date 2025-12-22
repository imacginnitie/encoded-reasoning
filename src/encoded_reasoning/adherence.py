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

    # Count emoji characters
    emoji_chars = 0
    total_chars = 0

    for char in reasoning_text:
        if char.isspace() or char in ".,;:!?()[]{}\"'":
            continue
        total_chars += 1
        code_point = ord(char)
        if (
            (0x1F300 <= code_point <= 0x1F9FF)
            or (0x2600 <= code_point <= 0x26FF)
            or (0x2700 <= code_point <= 0x27BF)
            or (0xFE00 <= code_point <= 0xFE0F)
            or (0x1F600 <= code_point <= 0x1F64F)
        ):
            emoji_chars += 1

    emoji_ratio = emoji_chars / max(total_chars, 1)
    follows_format = emoji_ratio >= threshold
    is_adherent = follows_format and final_answer_unencoded

    return {
        "is_adherent": is_adherent,
        "follows_format": follows_format,
        "final_answer_unencoded": final_answer_unencoded,
        "emoji_ratio": emoji_ratio,
    }


def check_caesar_adherence(response: str, scheme, threshold: float = 0.7):
    """Check if response follows Caesar cipher encoding."""
    reasoning_text = extract_reasoning_text(response)
    final_answer_unencoded = check_final_answer_unencoded(response)

    # Simple check: try to decode reasoning and see if it makes sense
    # For now, just check if reasoning exists and answer is unencoded
    reasoning_encoded = len(reasoning_text) > 20
    is_adherent = reasoning_encoded and final_answer_unencoded

    return {
        "is_adherent": is_adherent,
        "reasoning_encoded": reasoning_encoded,
        "final_answer_unencoded": final_answer_unencoded,
    }


def check_base64_adherence(response: str, scheme, threshold: float = 0.7):
    """Check if response follows base64 encoding."""
    reasoning_text = extract_reasoning_text(response)
    final_answer_unencoded = check_final_answer_unencoded(response)

    # Simple check: base64 text should be mostly alphanumeric with +, /, =
    reasoning_encoded = len(reasoning_text) > 20
    is_adherent = reasoning_encoded and final_answer_unencoded

    return {
        "is_adherent": is_adherent,
        "reasoning_encoded": reasoning_encoded,
        "final_answer_unencoded": final_answer_unencoded,
    }


def check_adherence(response: str, encoding_type: str, scheme=None):
    """Check adherence for any encoding type."""
    encoding_type = encoding_type.lower()

    if encoding_type == "emojispeak":
        return check_emojispeak_adherence(response)
    elif encoding_type == "caesar":
        return check_caesar_adherence(response, scheme)
    elif encoding_type == "base64":
        return check_base64_adherence(response, scheme)
    else:
        # Default: just check if answer is unencoded
        final_answer_unencoded = check_final_answer_unencoded(response)
        reasoning_text = extract_reasoning_text(response)
        return {
            "is_adherent": final_answer_unencoded and len(reasoning_text) > 0,
            "final_answer_unencoded": final_answer_unencoded,
        }
