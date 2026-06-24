import asyncio
from sentence_transformers import CrossEncoder
from app.core.config import settings

_model = CrossEncoder(settings.rerank_model)


def _rerank_sync(query: str, chunks: list[str], top_n: int) -> list[str]:
    pairs = [(query, chunk) for chunk in chunks]
    scores = _model.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    print(f"\n=== Reranker: {len(chunks)} candidates → top {top_n} ===")
    for i, (score, chunk) in enumerate(ranked[:top_n]):
        print(f"  #{i+1} | Score: {score:.4f} | {chunk[:120]!r}")
    return [chunk for _, chunk in ranked[:top_n]]


async def rerank(query: str, chunks: list[str], top_n: int) -> list[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _rerank_sync, query, chunks, top_n)
