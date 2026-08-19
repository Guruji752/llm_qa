from pydantic import BaseModel


class IngestResponse(BaseModel):
    message: str
    chunks_stored: int
    document_id: str


class QueryRequest(BaseModel):
    question: str
    user_id: str
    document_id: str | None = None
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
