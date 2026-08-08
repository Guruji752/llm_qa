import asyncio
import numpy as np
from huggingface_hub import InferenceClient
from langfuse import observe
from app.core.config import settings

_client = InferenceClient(token=settings.hf_token)

_MAX_CHARS = 1500


def _embed_one(text: str) -> list[float]:
    result = _client.feature_extraction(text[:_MAX_CHARS], model=settings.embedding_model)
    arr = np.array(result)
    # API returns token-level embeddings (seq_len, dim) — mean-pool to get sentence vector
    if arr.ndim == 1:
        return arr.tolist()
    elif arr.ndim == 2:
        return arr.mean(axis=0).tolist()
    else:
        return arr[0].mean(axis=0).tolist()


@observe(name="embed", as_type="embedding")
async def embed(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, _embed_one, t) for t in texts]
    return list(await asyncio.gather(*tasks))
