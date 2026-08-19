import uuid

from fastapi import APIRouter, UploadFile, File, Form
from app.core.config import settings
from app.models.schemas import IngestResponse
from app.services import pdf_processor, embedder, vector_store

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(user_id: str = Form(...), file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    document_id = str(uuid.uuid4())

    chunks = pdf_processor.extract_chunks(pdf_bytes, settings.chunk_size, settings.chunk_overlap)
    embeddings = await embedder.embed([chunk["text"] for chunk in chunks])
    await vector_store.ensure_collection(vector_size=len(embeddings[0]))
    await vector_store.upsert(
        chunks,
        embeddings,
        document_id=document_id,
        document_name=file.filename,
        user_id=user_id,
    )
    return IngestResponse(
        message="PDF ingested successfully",
        chunks_stored=len(chunks),
        document_id=document_id,
    )
