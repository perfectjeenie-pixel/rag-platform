import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.models import ChatRequest, ChatResponse, DocumentResponse
from app.security import require_api_key
from app.services import RagService
from app.text import extract_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@lru_cache
def get_rag_service() -> RagService:
    return RagService(get_settings())


app = FastAPI(title="Agentic RAG Platform", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/documents", response_model=DocumentResponse, dependencies=[Depends(require_api_key)])
async def upload_document(file: UploadFile = File(...), settings: Settings = Depends(get_settings)) -> DocumentResponse:
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File is too large")
    try:
        text = extract_text(file.filename or "upload.txt", content)
        return get_rag_service().index_document(file.filename or "upload.txt", text)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(request: ChatRequest) -> ChatResponse:
    return get_rag_service().answer(request.question, request.document_ids)