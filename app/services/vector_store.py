import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import settings

client = AsyncQdrantClient(url=settings.qdrant_url)


async def ensure_collection(vector_size: int) -> None:
    existing = [c.name for c in (await client.get_collections()).collections]
    if settings.qdrant_collection not in existing:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


async def upsert(chunks: list[str], embeddings: list[list[float]]) -> None:
    points = [
        PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_OID, chunk)), vector=vec, payload={"text": chunk})
        for chunk, vec in zip(chunks, embeddings)
    ]
    await client.upsert(collection_name=settings.qdrant_collection, points=points)


async def search(query_vector: list[float], top_k: int) -> list[str]:
    results = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
    )
    return [hit.payload["text"] for hit in results.points]
