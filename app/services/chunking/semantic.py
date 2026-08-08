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
from huggingface_hub import InferenceClient
from app.core.config import settings

_client = InferenceClient(token=settings.hf_token)


def semantic_chunks(
    text: str,
    threshold: float = 0.50,
) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    result = _client.feature_extraction(sentences, model=settings.embedding_model)
    embeddings = np.array(result).tolist()

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
    # Split on blank lines first (hard section boundaries), then on newlines
    lines = re.split(r"\n\s*\n|\n", text.strip())
    lines = [l.strip() for l in lines if l.strip()]

    # Merge lines that are too short (< 5 words) into the next line to avoid
    # fragmenting standalone date ranges or labels like "Experience"
    merged: list[str] = []
    buffer = ""
    for line in lines:
        if buffer:
            line = buffer + " " + line
            buffer = ""
        if len(line.split()) < 5 and merged:
            merged[-1] = merged[-1] + " " + line
        elif len(line.split()) < 5:
            buffer = line
        else:
            merged.append(line)

    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    return merged


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0
