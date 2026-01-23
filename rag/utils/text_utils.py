"""
Small text utilities.
"""
import re

def normalize_whitespace(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()

def short_preview(text: str, n: int = 300) -> str:
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0] + "..."
