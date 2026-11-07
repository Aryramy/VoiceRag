import io
from fastapi.testclient import TestClient
from app.main import app
from app.worker.audio_pipeline import strip_citations_for_tts

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "collection" in data

def test_documents_endpoint():
    res = client.get("/api/documents")
    assert res.status_code == 200
    data = res.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)

def test_upload_endpoint():
    dummy_text = b"# Dev Ingestion\nThis is a sample document for testing pre-flight upload."
    res = client.post(
        "/api/upload",
        files={"file": ("dev_test_doc.md", io.BytesIO(dummy_text), "text/markdown")}
    )
    assert res.status_code == 202
    data = res.json()
    assert data["status"] == "queued"
    assert data["filename"] == "dev_test_doc.md"

def test_chat_and_history_endpoints():
    session_id = "preflight-session-test"
    res = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "What is the return policy?"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "citations" in data
    assert data["session_id"] == session_id

    # Test history endpoint
    h_res = client.get(f"/api/chat/history/{session_id}")
    assert h_res.status_code == 200
    h_data = h_res.json()
    assert "history" in h_data
    assert len(h_data["history"]) >= 2

def test_citation_stripping():
    sample_text = "Here is an answer [chunk_123] and another fact [chunk_456]."
    cleaned = strip_citations_for_tts(sample_text)
    assert "[chunk_123]" not in cleaned
    assert "[chunk_456]" not in cleaned
    assert "Here is an answer and another fact." == cleaned.strip()

