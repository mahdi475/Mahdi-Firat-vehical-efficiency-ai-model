from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_examples_endpoint():
    response = client.get("/api/examples")

    assert response.status_code == 200
    assert len(response.json()) >= 3


def test_predict_validation_rejects_bad_payload():
    response = client.post(
        "/api/predict",
        json={
            "cylinders": 2,
            "displacement": 140.0,
            "horsepower": 90.0,
            "weight": 2264.0,
            "acceleration": 15.5,
            "model_year": 71,
            "origin": 2,
        },
    )

    assert response.status_code == 422
