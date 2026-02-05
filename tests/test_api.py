"""
Tests for the FastAPI prediction service
"""

import pytest
from fastapi.testclient import TestClient
from src.predict_api import app


client = TestClient(app)


class TestHealthEndpoint:
    """Test the /health endpoint"""

    def test_health_returns_200(self):
        """Health check should return 200"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self):
        """Health check should confirm model is loaded"""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    """Test the /predict endpoint"""

    def test_predict_returns_200(self):
        """Valid review should return 200"""
        response = client.post(
            "/predict",
            json={"review": "Great medication, worked perfectly!"}
        )
        assert response.status_code == 200

    def test_predict_response_format(self):
        """Response should have sentiment, confidence, probabilities"""
        response = client.post(
            "/predict",
            json={"review": "This drug helped my condition a lot."}
        )
        data = response.json()
        assert "sentiment" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["sentiment"] in ["positive", "negative", "neutral"]
        assert 0 <= data["confidence"] <= 1

    def test_predict_empty_review(self):
        """Empty review should return 400 error"""
        response = client.post(
            "/predict",
            json={"review": "   "}
        )
        assert response.status_code == 400

    def test_predict_positive_sentiment(self):
        """Clearly positive review should return positive"""
        response = client.post(
            "/predict",
            json={"review": "Best medicine ever! No side effects and I feel great!"}
        )
        data = response.json()
        assert data["sentiment"] == "positive"

    def test_predict_negative_sentiment(self):
        """Clearly negative review should return negative"""
        response = client.post(
            "/predict",
            json={"review": "Horrible drug. Caused severe pain and vomiting."}
        )
        data = response.json()
        assert data["sentiment"] == "negative"


class TestBatchEndpoint:
    """Test the /predict/batch endpoint"""

    def test_batch_predict(self):
        """Batch prediction should process multiple reviews"""
        reviews = [
            {"review": "Works great!"},
            {"review": "Terrible experience."},
            {"review": "It was okay, nothing special."}
        ]
        response = client.post("/predict/batch", json=reviews)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["predictions"]) == 3
