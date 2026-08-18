from pydantic import BaseModel, Field


class Source(BaseModel):
    """A source document chunk used in RAG retrieval."""

    document_id: str
    filename: str
    chunk_index: int
    text: str
    relevance_score: float = Field(default=0.0, description="Relevance score from 0 to 1")


class DocumentResponse(BaseModel):
    """Response after uploading and indexing a document."""

    document_id: str
    filename: str
    chunks_indexed: int


class ChatRequest(BaseModel):
    """User request for chat/QA."""

    question: str
    document_ids: list[str] | None = Field(default=None, description="Optional filter to specific documents")


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    used_retrieval: bool
