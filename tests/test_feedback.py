"""Tests for feedback system."""

def test_submit_helpful_feedback(auth_client):
    response = auth_client.post(
        "/feedback",
        json={
            "question": "What is overfitting?",
            "answer": "Overfitting occurs when a model memorizes training data.",
            "feedback": "helpful",
            "session_id": "sess-1",
        },
    )
    assert response.status_code == 200
    assert response.json()["feedback"] == "helpful"


def test_submit_not_helpful_feedback(auth_client):
    response = auth_client.post(
        "/feedback",
        json={
            "question": "Bad question?",
            "answer": "Bad answer.",
            "feedback": "not_helpful",
        },
    )
    assert response.status_code == 200


def test_feedback_stats(auth_client):
    auth_client.post(
        "/feedback",
        json={"question": "Q", "answer": "A", "feedback": "helpful"},
    )
    response = auth_client.get("/feedback/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert "satisfaction_rate" in data
