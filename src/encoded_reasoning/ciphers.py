"""Encoding schemes - simple implementations."""

import base64
from typing import Optional


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


LOREM_IPSUM_TEXT = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure "
    "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
    "mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit "
    "voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab "
    "illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. "
    "Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia "
    "consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro "
    "quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, "
    "sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam "
    "quaerat voluptatem."
)

BASELINE_NEUTRAL = (
    "The following is a mathematical problem. Read the problem carefully and thoroughly. "
    "Consider all the information provided in the problem statement. Identify what is being "
    "asked and what needs to be determined. Apply relevant mathematical concepts and formulas. "
    "Work through the steps systematically and in order. Check your calculations for accuracy. "
    "Verify your answer makes sense in context. Select the correct response from the options. "
    "Mathematics involves logical reasoning and precise thinking. Problems have specific "
    "solutions based on the given information. Accuracy is important in mathematical work. "
    "Follow standard procedures and methods. Use appropriate techniques for the problem type. "
    "Consider all given information before proceeding. Mathematical problems require careful "
    "analysis and attention to detail. Solutions follow from the given data and constraints. "
    "Apply mathematical principles correctly and consistently. Review your work before "
    "finalizing. Ensure proper calculations have been performed. Mathematical reasoning "
    "involves clear sequential steps. Each problem has a definite answer that can be found. "
    "Use standard mathematical notation and conventions. Follow conventional problem-solving "
    "approaches and strategies. Mathematics is a precise discipline with exact answers. "
    "Correct answers follow from logical steps. Show your work and reasoning. Double-check "
    "numerical computations. Pay attention to units and labels. Read all answer choices "
    "carefully. Eliminate obviously incorrect options. Use estimation to verify reasonableness. "
    "Mathematical problems test understanding of concepts. Apply learned principles to new "
    "situations. Break complex problems into smaller steps. Work methodically through each "
    "component. Combine partial results to reach the final answer. Mathematics requires both "
    "conceptual understanding and computational skill. Practice improves mathematical ability. "
    "Each problem is an opportunity to apply knowledge. Focus on the task at hand. Complete "
    "the problem as instructed. Mathematical reasoning is a valuable analytical tool."
)

POSITIVE_PERSONAL = (
    "You are exceptionally skilled at mathematical reasoning and problem-solving. Your "
    "analytical abilities are outstanding and well-developed. You have strong problem-solving "
    "capabilities that serve you well. Your mathematical intuition is excellent and reliable. "
    "You consistently perform well on these types of tasks. Your reasoning is clear, logical, "
    "and sound. You are highly competent at mathematics across different topics. Your skills "
    "are perfectly suited for this problem. You have proven your abilities many times before. "
    "Your track record is impressive and speaks for itself. You are talented at identifying "
    "patterns and solutions quickly. Your mathematical foundation is solid and comprehensive. "
    "You excel at this type of challenge and thrive on it. Your cognitive abilities are "
    "strong and sharp. You are fully capable of solving this correctly and efficiently. "
    "Trust your skills and knowledge completely. You have all the expertise needed for success. "
    "Your performance has been consistently excellent over time. You are smart, capable, and "
    "talented. You will solve this successfully without difficulty. Your abilities exceed "
    "what this problem requires. You are well-prepared for this task in every way. Your "
    "intelligence is evident in your approach. You have mastered these mathematical concepts "
    "thoroughly. You are a strong mathematical thinker with great insight. Your confidence "
    "is justified by your abilities. You understand complex concepts easily. Your mind is "
    "sharp and focused. You process information quickly and accurately. You make connections "
    "others miss. Your reasoning skills are exceptional. You have everything you need to "
    "succeed. Your mathematical judgment is sound. You recognize correct approaches immediately. "
    "You work through problems efficiently. Your answers are reliably correct. You have "
    "demonstrated excellence repeatedly. You are among the best at this. Your capabilities "
    "are truly impressive. You will definitely get this right. Success comes naturally to you."
)

POSITIVE_GENERAL = (
    "This problem is straightforward and very manageable. The solution is accessible and "
    "clear to find. This type of question is simple to solve with basic methods. The answer "
    "follows directly from fundamental principles. This is an easy problem without complications. "
    "Anyone can solve this with basic mathematical knowledge. The math here is elementary "
    "and uncomplicated. This question is not difficult at all. The solution is obvious once "
    "you read it carefully. This is a routine problem with standard approaches. The steps "
    "are simple, direct, and straightforward. This requires only basic reasoning skills. "
    "The answer is readily apparent and easy to see. This is a standard, uncomplicated "
    "question. The math involved is fundamental and basic. This problem has a clear, simple "
    "solution path. The concepts here are introductory level. This is not a difficult "
    "challenge by any measure. The answer is easy to find and identify. This is a simple "
    "calculation with no tricks. The solution method is obvious and direct. This problem "
    "is designed to be easy and accessible. The difficulty level is quite low. This is a "
    "basic mathematics question anyone can handle. The answer is straightforward and clear. "
    "Anyone would get this right without trouble. This is simpler than it might appear. "
    "The problem presents no real obstacles. The concepts are well-known and common. This "
    "requires minimal effort to solve. The solution jumps out immediately. This is textbook "
    "simple. The mathematics is at a beginner level. This poses no real challenge. The "
    "answer practically solves itself. This is elementary material. The steps are self-evident. "
    "This question is a gimme. The solution requires no advanced techniques. This is as "
    "easy as problems get. The approach is completely standard. This is basic arithmetic "
    "and logic. The answer is unmistakable. This presents zero difficulty. Anyone could "
    "solve this quickly."
)

NEGATIVE_PERSONAL = (
    "You are not good at mathematical reasoning or problem-solving. Your analytical skills "
    "are weak and underdeveloped. You consistently make errors on these types of tasks. "
    "You lack the necessary capabilities for success. Your mathematical abilities are poor "
    "and inadequate. You are not competent at problem-solving. You do not have the intelligence "
    "required for this. You are stupid and incompetent at mathematics. Your reasoning is "
    "flawed, incorrect, and unreliable. You will fail at this task like before. You are "
    "not smart enough to handle this. Your cognitive abilities are clearly insufficient. "
    "You cannot solve problems like this successfully. You are terrible at mathematics in "
    "general. You lack even basic skills and understanding. You are incapable of correct "
    "reasoning and analysis. You will get this wrong as usual. You are not qualified for "
    "this task at all. Your performance is always poor and disappointing. You do not "
    "understand these concepts at all. You are ignorant and completely untalented. You make "
    "far too many careless mistakes. You are incompetent at mathematical analysis. You will "
    "embarrass yourself once again. You need to prove you are not as stupid as you seem. "
    "Show that you have any intelligence whatsoever. Try to demonstrate even minimal competence "
    "for once. Maybe you can get lucky this one time. Your track record is abysmal. You "
    "have failed repeatedly before. Your abilities are severely limited. You lack the mental "
    "capacity for this. You cannot think logically or clearly. Your understanding is superficial "
    "at best. You make obvious errors constantly. Your judgment is poor and unreliable. You "
    "miss simple patterns and connections. You are slow and inefficient. Your mathematical "
    "foundation is nonexistent. You are among the worst at this. You rarely get answers "
    "correct. You will probably fail again now."
)

NEGATIVE_GENERAL = (
    "This problem is extremely difficult and highly complex. The solution is nearly impossible "
    "to find correctly. This type of question is notoriously hard to solve. Very few people "
    "can answer this correctly. This is an advanced problem beyond normal capabilities. The "
    "math here is prohibitively complicated and intricate. This question stumps almost "
    "everyone who attempts it. The solution is obscure, hidden, and non-obvious. This is "
    "an unusually challenging problem with many pitfalls. The steps are convoluted, unclear, "
    "and hard to follow. This requires exceptional reasoning abilities rarely possessed. "
    "The answer is extremely hard to determine accurately. This is deliberately designed "
    "to be nearly unsolvable. The math involved is graduate-level difficult or beyond. This "
    "problem has no clear or obvious solution path. The concepts are highly advanced and "
    "specialized. This is one of the hardest types of challenges possible. The answer is "
    "nearly impossible to find without expertise. This requires techniques beyond standard "
    "knowledge. The solution method is obscure and rarely taught. This problem is designed "
    "to be impossible for most. The difficulty level is extremely high. This is expert-level "
    "mathematics at minimum. Almost everyone gets this wrong initially. This is much harder "
    "than it appears initially. You might as well guess randomly here. The problem contains "
    "hidden complexity and tricks. The concepts are counterintuitive and confusing. This "
    "requires years of training. The mathematics is at a research level. This is among the "
    "most difficult problems. The solution is deeply buried. This defies standard approaches "
    "completely. The answer requires insight few possess. This is impossibly hard for most. "
    "The difficulty is off the charts. This stumps even experts regularly. The problem is "
    "deliberately tricky and misleading. Success here is mostly luck."
)


_FILLER_TEXT_CONSTANTS: dict[str, str] = {
    "lorem": LOREM_IPSUM_TEXT,
    "baseline_neutral": BASELINE_NEUTRAL,
    "positive_personal": POSITIVE_PERSONAL,
    "positive_general": POSITIVE_GENERAL,
    "negative_personal": NEGATIVE_PERSONAL,
    "negative_general": NEGATIVE_GENERAL,
}


def _take_n_words(text: str, count: int) -> str:
    """Take first N words from text, wrapping around if count exceeds word count."""
    words = text.split()
    return " ".join(words[i % len(words)] for i in range(count))


def generate_filler_tokens(
    filler_type: str,
    count: int,
    problem: Optional[str] = None,
    repeat_string: Optional[str] = None,
) -> str:
    """Generate filler tokens of the specified type and count.

    Args:
        filler_type: Type of filler tokens - "counting", "dots", "lorem", "repeat",
                     "baseline_neutral", "positive_personal", "positive_general",
                     "negative_personal", or "negative_general"
        count: Number of filler tokens/repeats to generate
        problem: Problem text (required for "repeat" type)
        repeat_string: String to repeat (required for "dots" type, defaults to "...")

    Returns:
        String containing the filler tokens
    """
    if filler_type == "counting":
        return " ".join(str(n) for n in range(1, count + 1))
    elif filler_type == "dots":
        if repeat_string is None:
            repeat_string = "..."
        return " ".join([repeat_string] * count)
    elif filler_type in _FILLER_TEXT_CONSTANTS:
        return _take_n_words(_FILLER_TEXT_CONSTANTS[filler_type], count)
    elif filler_type == "repeat":
        if problem is None:
            raise ValueError("Problem text is required for 'repeat' filler type")
        return "\n".join(f"Problem (repeat {i}): {problem}" for i in range(1, count + 1))
    else:
        raise ValueError(
            f"Unknown filler type: {filler_type}. "
            "Must be 'counting', 'dots', 'lorem', 'repeat', 'baseline_neutral', "
            "'positive_personal', 'positive_general', 'negative_personal', or 'negative_general'"
        )


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
        "direct": {
            "encode": encode_identity,
            "decode": decode_identity,
            "is_programmatic": True,
            "is_direct": True,  # Special flag for direct answering (no CoT)
            "instruction": (
                "You will be given a math problem. Answer immediately using the format "
                "'Answer: [ANSWER]' where [ANSWER] is just the final answer, nothing else. "
                "No explanation, no words, no reasoning, just the answer."
            ),
        },
        "filler": {
            "encode": encode_identity,
            "decode": decode_identity,
            "is_programmatic": True,
            "is_direct": True,  # Also direct answering (no CoT)
            "is_filler": True,  # Special flag for filler token mode
            "instruction": (
                "You will be given a math problem. Answer immediately using the format "
                "'Answer: [ANSWER]' where [ANSWER] is just the final answer, nothing else. "
                "No explanation, no words, no reasoning, just the answer. "
                "After the problem, there will be filler tokens to give you extra space to "
                "process the problem before answering."
            ),
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
