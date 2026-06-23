"""Tests for JWT authentication."""

import pytest


def test_register_user(auth_client):
    import uuid

    uid = uuid.uuid4().hex[:8]
    response = auth_client.post(
        "/auth/register",
        json={
            "email": f"student_{uid}@rgpv.edu",
            "username": f"student_{uid}",
            "password": "securepass123",
            "full_name": "RGPV Student",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert "@rgpv.edu" in data["email"]


def test_register_duplicate_email(auth_client):
    payload = {
        "email": "dup@rgpv.edu",
        "username": "user_a",
        "password": "securepass123",
    }
    auth_client.post("/auth/register", json=payload)
    response = auth_client.post(
        "/auth/register",
        json={**payload, "username": "user_b"},
    )
    assert response.status_code == 400


def test_login_success(auth_client):
    auth_client.post(
        "/auth/register",
        json={
            "email": "login@rgpv.edu",
            "username": "loginuser",
            "password": "securepass123",
        },
    )
    response = auth_client.post(
        "/auth/login",
        json={"email": "login@rgpv.edu", "password": "securepass123"},
    )
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_login_invalid_credentials(auth_client):
    response = auth_client.post(
        "/auth/login",
        json={"email": "nobody@rgpv.edu", "password": "wrong"},
    )
    assert response.status_code == 401


def test_me_endpoint(auth_client):
    response = auth_client.get("/auth/me")
    assert response.status_code == 200
    assert response.json()["email"] == "test@rgpv.edu"


def test_logout(auth_client):
    response = auth_client.post("/auth/logout")
    assert response.status_code == 200
