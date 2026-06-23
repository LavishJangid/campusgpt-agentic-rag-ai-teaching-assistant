"""Tests for RAGAS evaluation endpoint."""

def test_evaluation_endpoint(auth_client):
    response = auth_client.post(
        "/evaluation",
        json={"question": "What is data science?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "faithfulness" in data
    assert "context_precision" in data
    assert "answer_relevancy" in data
    assert "latency_seconds" in data
    assert "answer" in data


def test_evaluation_history(auth_client):
    auth_client.post("/evaluation", json={"question": "Define machine learning"})
    response = auth_client.get("/evaluation/history")
    assert response.status_code == 200
    assert "runs" in response.json()
