import asyncio
from huggingface_hub import InferenceClient
from langfuse import observe
from app.core.config import settings

_client = InferenceClient(token=settings.hf_token)


def _rerank_sync(query: str, chunks: list[str], top_n: int) -> list[str]:
    results = _client.text_ranking(text=query, text_pair=chunks, model=settings.rerank_model)
    ranked = sorted(zip(results, chunks), key=lambda x: x[0].score, reverse=True)
    print(f"\n=== Reranker: {len(chunks)} candidates → top {top_n} ===")
    for i, (result, chunk) in enumerate(ranked[:top_n]):
        print(f"  #{i+1} | Score: {result.score:.4f} | {chunk[:120]!r}")
    return [chunk for _, chunk in ranked[:top_n]]


@observe(name="rerank", as_type="tool")
async def rerank(query: str, chunks: list[str], top_n: int) -> list[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _rerank_sync, query, chunks, top_n)
