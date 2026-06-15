from fastapi import APIRouter
from app.models.schemas import QueryRequest, QueryResponse
from app.services import embedder, vector_store, llm

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    query_embedding = await embedder.embed([request.question])
    context_chunks = await vector_store.search(query_embedding[0], request.top_k)
    answer = await llm.generate_answer(request.question, context_chunks)
    return QueryResponse(answer=answer, sources=context_chunks)
