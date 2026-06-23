"""Additional API endpoint tests."""

def test_search_endpoints(auth_client):
    for path in [
        "/search/topic/Normalization",
        "/search/subject/DBMS?query=SQL",
        "/search/semester/5?query=units",
    ]:
        r = auth_client.get(path)
        assert r.status_code == 200
        assert "count" in r.json()


def test_chat_validation_errors(auth_client):
    r = auth_client.post("/chat", json={"question": "   ", "session_id": "x"})
    assert r.status_code == 400

    r = auth_client.post("/chat", json={"question": "x" * 5001, "session_id": "x"})
    assert r.status_code == 422  # Pydantic max_length validation
