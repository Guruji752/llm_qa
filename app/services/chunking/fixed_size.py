"""
Fixed-size sliding window chunking.

Splits text into chunks of exactly `size` characters, advancing by
`size - overlap` each step. Fast and simple, but cuts across sentence
and paragraph boundaries.

When to use: baseline / prototyping. Swap out once retrieval quality matters.
"""


def fixed_size_chunks(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks
