from pydantic import BaseModel


class IngestResponse(BaseModel):
    message: str
    chunks_stored: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
