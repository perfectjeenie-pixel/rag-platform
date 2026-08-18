from uuid import uuid4

from openai import OpenAI
from qdrant_client import models

from app.config import Settings
from app.models import ChatResponse, DocumentResponse, Source
from app.text import split_text
from app.vector_store import VectorStore, point_id


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.store = VectorStore(settings)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.openai.embeddings.create(model=self.settings.openai_embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def index_document(self, filename: str, text: str) -> DocumentResponse:
        chunks = split_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("No readable text was found in this document")
        document_id = str(uuid4())
        vectors = self._embed(chunks)
        self.store.ensure_collection(len(vectors[0]))
        points = [
            models.PointStruct(
                id=point_id(document_id, index),
                vector=vector,
                payload={"document_id": document_id, "filename": filename, "chunk_index": index, "text": chunk},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self.store.upsert(points)
        return DocumentResponse(document_id=document_id, filename=filename, chunks_indexed=len(chunks))

    def _needs_retrieval(self, question: str) -> bool:
        prompt = "Reply with only YES or NO. Does this question require information from a private knowledge base?\nQuestion: " + question
        result = self.openai.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0,
            max_tokens=3,
            messages=[{"role": "user", "content": prompt}],
        )
        return result.choices[0].message.content.strip().upper().startswith("Y")

    def answer(self, question: str, document_ids: list[str] | None = None) -> ChatResponse:
        use_retrieval = self._needs_retrieval(question)
        sources: list[Source] = []
        context = ""
        if use_retrieval:
            sources = self.store.search(self._embed([question])[0], self.settings.top_k, document_ids)
            context = "\n\n".join(f"[Source {i + 1}] {source.text}" for i, source in enumerate(sources))

        system = (
            "You are a precise assistant. Answer using the provided sources when available. "
            "Do not invent policies or facts. If the sources do not support an answer, say so. "
            "Use inline citations like [Source 1]."
        )
        user = f"Question: {question}\n\nSources:\n{context or 'No retrieved sources.'}"
        completion = self.openai.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0.2,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return ChatResponse(
            answer=completion.choices[0].message.content or "I could not generate an answer.",
            sources=sources,
            used_retrieval=use_retrieval,
        )