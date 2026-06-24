import logging

from fastapi import APIRouter
from app.models.schemas import QueryRequest, QueryResponse
from app.services import embedder, vector_store, llm, reranker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    query_embedding = await embedder.embed([request.question])
    candidates = await vector_store.search(query_embedding[0], request.top_k * 3)
    context_chunks = await reranker.rerank(request.question, candidates, request.top_k)
    answer = await llm.generate_answer(request.question, context_chunks)
    return QueryResponse(answer=answer, sources=context_chunks)
