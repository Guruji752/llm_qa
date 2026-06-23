import fitz  # pymupdf

from app.services.chunking import fixed_size_chunks,recursive_chunks



def extract_chunks(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "\n".join(page.get_text() for page in doc)
    return recursive_chunks(full_text, size=chunk_size, overlap=chunk_overlap)
