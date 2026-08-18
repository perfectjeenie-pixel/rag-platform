import io
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_without_api_key():
    """Test that upload requires API key."""
    file_content = b"Test content"
    response = client.post(
        "/v1/documents",
        files={"file": ("test.txt", file_content)},
    )
    assert response.status_code == 403


def test_chat_without_api_key():
    """Test that chat requires API key."""
    response = client.post(
        "/v1/chat",
        json={"question": "What is RAG?"},
    )
    assert response.status_code == 403
