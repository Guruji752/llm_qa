"""
Semantic chunking.

Embeds every sentence, then measures cosine similarity between adjacent
sentences. A new chunk starts wherever similarity drops below `threshold`
(a topic boundary). Produces variable-length chunks that align with actual
topic shifts rather than arbitrary character counts.

This is the most retrieval-accurate strategy but also the most expensive —
it makes one embedding call per sentence. Only worth it when chunk quality
is the bottleneck.

Requires: openai (already in your stack via embedder.py)
"""

import re

import numpy as np
from openai import OpenAI


def semantic_chunks(
    text: str,
    threshold: float = 0.75,
    embed_model: str = "text-embedding-3-small",
    client: OpenAI | None = None,
) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    client = client or OpenAI()
    embeddings = _embed_batch(sentences, embed_model, client)

    chunks: list[str] = []
    group = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        if sim >= threshold:
            group.append(sentences[i])
        else:
            chunks.append(" ".join(group))
            group = [sentences[i]]

    if group:
        chunks.append(" ".join(group))

    return chunks


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _embed_batch(sentences: list[str], model: str, client: OpenAI) -> list[list[float]]:
    response = client.embeddings.create(input=sentences, model=model)
    return [item.embedding for item in response.data]


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0
