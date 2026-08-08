import logging

from fastapi import APIRouter
from app.models.schemas import QueryRequest, QueryResponse
from app.services import embedder, vector_store, llm, reranker
# from langfuse.decorators import observe, langfuse_context
from langfuse import observe

from langfuse import Langfuse
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
langfuse = Langfuse(public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_BASE_URL)


@router.post("/query", response_model=QueryResponse)
@observe()
async def query(request: QueryRequest):
    query_embedding = await embedder.embed([request.question])
    candidates = await vector_store.search(query_embedding[0], request.top_k * 3)
    # context_chunks = await reranker.rerank(request.question, candidates, request.top_k)
    # candidates = await vector_store.search(query_embedding[0], request.top_k)
    answer = await llm.generate_answer(request.question, candidates)
    return QueryResponse(answer=answer, sources=candidates)
