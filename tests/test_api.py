from fastapi.testclient import TestClient
from app.main import app

def test_fastapi_endpoints():
    client = TestClient(app)
    
    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    data = res.json()
    print("Health Check Response:", data)
    assert data["status"] == "ok"

    # 2. Join token endpoint
    res = client.get("/api/get-join-token?room_name=test-room&participant_name=test-agent")
    assert res.status_code == 200, f"Token generation failed: {res.text}"
    token_data = res.json()
    print("Token Endpoint Response:")
    print(f"  URL: {token_data.get('url')}")
    print(f"  Room: {token_data.get('room')}")
    print(f"  Participant: {token_data.get('participant')}")
    print(f"  Token length: {len(token_data.get('token', ''))}")
    assert "token" in token_data and len(token_data["token"]) > 20

    print("=== All FastAPI Gateway tests passed! ===")

if __name__ == "__main__":
    test_fastapi_endpoints()

