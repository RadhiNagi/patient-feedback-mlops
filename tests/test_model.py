"""
Tests for model training and prediction
"""

import pytest
import os
import json
import joblib
import numpy as np


class TestModelArtifacts:
    """Test that model files exist and are valid"""

    def test_model_file_exists(self):
        """Trained model file should exist"""
        assert os.path.exists("models/sentiment_model.joblib"), \
            "Model file not found. Run training first."

    def test_vectorizer_file_exists(self):
        """TF-IDF vectorizer file should exist"""
        assert os.path.exists("models/tfidf_vectorizer.joblib"), \
            "Vectorizer file not found. Run training first."

    def test_metrics_file_exists(self):
        """Metrics JSON file should exist"""
        assert os.path.exists("models/metrics.json"), \
            "Metrics file not found. Run training first."

    def test_metrics_format(self):
        """Metrics should contain required fields"""
        with open("models/metrics.json") as f:
            metrics = json.load(f)
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert "f1_per_class" in metrics

    def test_model_accuracy_above_threshold(self):
        """Model accuracy must be above 70% to deploy"""
        with open("models/metrics.json") as f:
            metrics = json.load(f)
        assert metrics["accuracy"] >= 0.70, \
            f"Accuracy {metrics['accuracy']} is below 70% threshold!"


class TestModelPrediction:
    """Test that model makes correct predictions"""

    @pytest.fixture
    def model_and_vectorizer(self):
        """Load model and vectorizer for testing"""
        model = joblib.load("models/sentiment_model.joblib")
        vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
        return model, vectorizer

    def test_positive_review(self, model_and_vectorizer):
        """Clearly positive review should predict positive (2)"""
        model, vectorizer = model_and_vectorizer
        text = "This drug is amazing! It completely cured my condition."
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]
        assert prediction == 2, f"Expected positive (2), got {prediction}"

    def test_negative_review(self, model_and_vectorizer):
        """Clearly negative review should predict negative (0)"""
        model, vectorizer = model_and_vectorizer
        text = "Terrible side effects. Made me very sick. Never again."
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]
        assert prediction == 0, f"Expected negative (0), got {prediction}"

    def test_prediction_returns_valid_label(self, model_and_vectorizer):
        """Predictions should only be 0, 1, or 2"""
        model, vectorizer = model_and_vectorizer
        text = "Some random review about medication"
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]
        assert prediction in [0, 1, 2], f"Invalid label: {prediction}"

    def test_predict_proba_sums_to_one(self, model_and_vectorizer):
        """Prediction probabilities should sum to approximately 1.0"""
        model, vectorizer = model_and_vectorizer
        text = "This medication helped with my pain"
        features = vectorizer.transform([text])
        probabilities = model.predict_proba(features)[0]
        assert abs(sum(probabilities) - 1.0) < 0.01, \
            f"Probabilities sum to {sum(probabilities)}, expected ~1.0"