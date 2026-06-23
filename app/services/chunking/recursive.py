"""
Recursive Character Splitting — how it works
---------------------------------------------
Goal: keep chunks under `size` chars while cutting at the most natural
boundary available, not at an arbitrary character position.

Algorithm (LangChain's RecursiveCharacterTextSplitter):

  1. Try the first separator ("\n\n" — paragraph break).
     Split the whole text on it.
     Any piece that is already <= size? Keep it as a chunk.
     Any piece still too big? Move to step 2 with just that piece.

  2. Try the next separator ("\n" — line break) on the oversized piece.
     Same rule: keep what fits, recurse on what doesn't.

  3. Continue down the list: ". " → " " → "" (character-by-character).

  4. Once all pieces are <= size, merge adjacent small pieces back together
     (greedy packing) so we don't produce hundreds of tiny chunks.

  5. Apply overlap: the tail of each chunk is prepended to the next one
     so a sentence split across a boundary still appears fully in one chunk.

Separator priority: "\n\n" > "\n" > ". " > " "
The algorithm always prefers the highest-level boundary it can use.

Example with size=30, overlap=5, text = "Hello world.\n\nGoodbye world.":

  Split on "\n\n"  → ["Hello world.", "Goodbye world."]
  Both <= 30       → no recursion needed
  After overlap    → ["Hello world.", "orld.Goodbye world."]

When to use: general-purpose default for most RAG pipelines.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_chunks(
    text: str,
    size: int = 500,
    overlap: int = 100,
    separators: list[str] | None = None,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=separators or ["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)
