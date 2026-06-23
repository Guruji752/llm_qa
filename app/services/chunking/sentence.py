"""
Sentence-boundary chunking.

Groups sentences into chunks so each chunk stays within `max_sentences`.
Overlap is expressed in sentences, not characters.

Splits on ". ", "! ", "? " — simple regex, no NLP library required.
Accurate enough for English prose; breaks on abbreviations (e.g. "Dr. Smith").

When to use: Q&A over documents where answers fit in 1-3 sentences.
"""

import re


def sentence_chunks(
    text: str,
    max_sentences: int = 5,
    overlap_sentences: int = 1,
) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    step = max(1, max_sentences - overlap_sentences)
    start = 0

    while start < len(sentences):
        group = sentences[start : start + max_sentences]
        chunks.append(" ".join(group))
        start += step

    return chunks


def _split_sentences(text: str) -> list[str]:
    # Split after ". ", "! ", "? " while keeping the punctuation on the left side.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
