import re


def clean_text(text: str) -> str:

    text = str(text)

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()