"""
Paragraph-boundary chunking.

Splits on blank lines (\n\n), then groups paragraphs into chunks that stay
under `max_chars`. Overlap carries the last paragraph of the previous chunk
into the next one.

When to use: structured documents (reports, papers, books) where paragraphs
are coherent units. Better retrieval than fixed-size at almost zero cost.
"""


def paragraph_chunks(
    text: str,
    max_chars: int = 1000,
    overlap_paragraphs: int = 1,
) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        # A single paragraph already exceeds the limit — emit it alone.
        if len(para) > max_chars:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
            chunks.append(para)
            current_parts = []
            current_len = 0
            continue

        if current_len + len(para) > max_chars and current_parts:
            chunks.append("\n\n".join(current_parts))
            # carry overlap paragraphs into the next chunk
            current_parts = current_parts[-overlap_paragraphs:] if overlap_paragraphs else []
            current_len = sum(len(p) for p in current_parts)

        current_parts.append(para)
        current_len += len(para)

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks
