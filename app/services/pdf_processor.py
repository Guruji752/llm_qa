import fitz  # pymupdf
import pymupdf4llm

from app.services.chunking import section_chunks


def extract_chunks(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # Markdown, not raw text: pymupdf4llm detects headings from actual font
    # size/bold, which plain page.get_text() throws away. section_chunks
    # relies on that structure to find real section boundaries.
    markdown_text = pymupdf4llm.to_markdown(doc)
    return section_chunks(markdown_text, max_chars=chunk_size)
