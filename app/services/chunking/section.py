"""
Section-based chunking (resume-specific), built on LangChain's
MarkdownHeaderTextSplitter.

Takes markdown produced by pymupdf4llm (see pdf_processor.py), not raw PDF
text. Plain text extraction throws away the one signal that actually marks a
heading in a resume — font size / bold — so a prior version of this chunker
guessed headings from line shape and got it wrong on anything that didn't
look like "Experience" sitting alone on its own line. pymupdf4llm does that
detection properly, using real font metrics, and emits markdown `#`/`##`
headings for whatever it identifies as visually distinct.

The catch MarkdownHeaderTextSplitter can't handle on its own: pymupdf4llm's
heading levels aren't reliable section boundaries. It promotes bold sub-lines
— a job title, a company name — to the same `##` level as an actual section
header like "Experience", because both are just bold text to it. Splitting
on raw heading level would fragment "Experience" into one orphan chunk per
job title. So before handing off to MarkdownHeaderTextSplitter, every
heading is normalized against a known-alias table: a real section name
("Work Experience" -> "Experience") is rewritten to a canonical `##`
heading; anything else (a job title, a company name) is demoted to plain
text so it folds into whichever section is currently open.

Each chunk is returned as a dict — not a bare string, unlike the other
chunkers here — because the section name IS the retrieval-relevant metadata:
{"section": "Experience", "text": "...", "part": 1}

A section that exceeds `max_chars` (Experience and Projects usually do) is
split further with RecursiveCharacterTextSplitter, with each part keeping
the same section name so retrieval on "Experience" still finds all of it.

When to use: resumes and other clearly-sectioned documents. Not useful for
free-flowing prose with no headings — everything falls into the single
leading "Header" bucket.
"""

import re

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")
_MD_EMPHASIS_RE = re.compile(r"^[*_]+|[*_]+$")

_SECTION_METADATA_KEY = "Section"
_HEADERS_TO_SPLIT_ON = [("##", _SECTION_METADATA_KEY)]

# Canonical section name -> header aliases that map to it. Matching is
# case-insensitive and ignores a trailing colon.
SECTION_ALIASES: dict[str, str] = {
    "summary": "Summary",
    "professional summary": "Summary",
    "objective": "Summary",
    "career objective": "Summary",
    "profile": "Summary",
    "experience": "Experience",
    "work experience": "Experience",
    "professional experience": "Experience",
    "employment history": "Experience",
    "work history": "Experience",
    "education": "Education",
    "academic background": "Education",
    "skills": "Skills",
    "technical skills": "Skills",
    "core competencies": "Skills",
    "key skills": "Skills",
    "projects": "Projects",
    "personal projects": "Projects",
    "academic projects": "Projects",
    "certifications": "Certifications",
    "certificates": "Certifications",
    "licenses": "Certifications",
    "achievements": "Achievements",
    "awards": "Achievements",
    "honors": "Achievements",
    "publications": "Publications",
    "languages": "Languages",
    "interests": "Interests",
    "hobbies": "Interests",
    "references": "References",
    "contact": "Contact",
    "contact information": "Contact",
    "volunteer experience": "Volunteering",
    "volunteering": "Volunteering",
    "extracurricular activities": "Extracurricular",
}


def section_chunks(
    markdown_text: str,
    max_chars: int = 1000,
    overlap: int = 100,
) -> list[dict]:
    normalized = _normalize_headings(markdown_text)

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS_TO_SPLIT_ON)
    sections = header_splitter.split_text(normalized)

    sub_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=overlap)

    chunks: list[dict] = []
    for doc in sections:
        content = doc.page_content.strip()
        if not content:
            continue
        name = doc.metadata.get(_SECTION_METADATA_KEY, "Header")
        parts = sub_splitter.split_text(content) if len(content) > max_chars else [content]
        for i, part_text in enumerate(parts, start=1):
            chunks.append({"section": name, "text": part_text, "part": i})

    return chunks


def _normalize_headings(markdown_text: str) -> str:
    lines = []
    for line in markdown_text.split("\n"):
        heading = _extract_heading(line)
        if heading is None:
            lines.append(line)
            continue

        canonical = SECTION_ALIASES.get(heading.rstrip(":").strip().lower())
        if canonical:
            lines.append(f"## {canonical}")
        else:
            # A bold/large line pymupdf4llm treated as a heading, but that
            # isn't a known resume section (a job title, a company name) —
            # demote it to plain text so it stays inside the open section.
            lines.append(heading)

    return "\n".join(lines)


def _extract_heading(line: str) -> str | None:
    match = _HEADING_RE.match(line.strip())
    if not match:
        return None
    text = _MD_EMPHASIS_RE.sub("", match.group(1).strip()).strip()
    return text or None
