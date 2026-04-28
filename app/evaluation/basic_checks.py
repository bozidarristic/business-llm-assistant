def has_insufficient_context_answer(answer: str) -> bool:
    return "do not have enough information" in answer.lower()


def contains_unwanted_guessing(answer: str) -> bool:
    risky_phrases = [
        "probably",
        "i assume",
        "it seems likely",
        "not mentioned but",
    ]
    return any(phrase in answer.lower() for phrase in risky_phrases)
