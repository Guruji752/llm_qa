from fastapi import FastAPI
from app.api.routes import ingest, query

app = FastAPI(title="RAG Service")

app.include_router(ingest.router)
app.include_router(query.router)
