import io
from pypdf import PdfReader


def extract_chunks(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int) -> list[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return _sliding_window(full_text, chunk_size, chunk_overlap)


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks
