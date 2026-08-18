# Agentic RAG Platform

A production-minded starter for a Retrieval-Augmented Generation (RAG) service. Upload documents, split and embed them, then ask grounded questions through an agentic retrieval workflow.

## What it includes

- FastAPI REST API with OpenAPI docs
- PDF, Markdown, and text ingestion
- Qdrant vector search with OpenAI embeddings
- Agent workflow that decides whether a question needs retrieval, fetches evidence, and generates a cited answer
- Docker Compose for the API and Qdrant
- API-key protection, structured logging, health checks, and tests

## Architecture

```text
Client -> FastAPI -> Ingestion / Chat service -> OpenAI
                              |                 |
                              v                 v
                         Qdrant vectors <- embeddings / answer
```

## Quick start

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Start the stack:

   ```bash
   docker compose up --build
   ```

3. Open `http://localhost:8000/docs`.

Or run locally:

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows PowerShell
pip install -e ".[dev]"
docker compose up qdrant -d
uvicorn app.main:app --reload
```

## API examples

Upload a document:

```bash
curl -X POST http://localhost:8000/v1/documents \
  -H "X-API-Key: change-me" \
  -F "file=@handbook.pdf"
```

Ask a question:

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{"question":"What is the leave policy?"}'
```

The answer includes `sources`, each with document name, chunk text, and a relevance score. Use the optional `document_ids` request field to restrict retrieval to specific documents.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI credential | required |
| `APP_API_KEY` | API key passed in `X-API-Key` | `change-me` |
| `QDRANT_URL` | Qdrant endpoint | `http://localhost:6333` |
| `QDRANT_COLLECTION` | Vector collection name | `knowledge_base` |
| `OPENAI_CHAT_MODEL` | Answer generation model | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | Embeddings model | `text-embedding-3-small` |

## Production notes

- Put authentication behind your identity provider instead of using the sample static API key.
- Store original documents in object storage and add a persistent metadata database.
- Add tenant IDs to metadata and filters before serving multiple customers.
- Add rate limiting, observability, background job queues, and evaluation datasets before production rollout.

## Tests

```bash
pytest
```

## Repository hygiene

The repository intentionally excludes `.env`, virtual environments, local vector data, uploads, and test caches. Never commit credentials.