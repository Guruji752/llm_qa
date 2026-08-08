import fitz  # pymupdf

from app.services.chunking import paragraph_chunks


def extract_chunks(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "\n".join(page.get_text() for page in doc)
    return paragraph_chunks(full_text, max_chars=chunk_size)
