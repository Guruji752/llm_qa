import asyncio
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.core.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


async def embed(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    # SentenceTransformer is CPU-bound and synchronous, so we offload it
    # to a thread to avoid blocking FastAPI's async event loop
    vectors = await loop.run_in_executor(None, _get_model().encode, texts)
    return vectors.tolist()
