"""
FastAPI Prediction Service
Serves the trained sentiment model as a REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

# Initialize FastAPI app
app = FastAPI(
    title="Drug Review Sentiment API",
    description="Predicts sentiment (positive/negative/neutral) from patient drug reviews",
    version="1.0.0"
)

# Load model and vectorizer at startup
MODEL_DIR = os.getenv("MODEL_DIR", "models")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    metrics = None
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vectorizer = None
    metrics = None


# Request/Response schemas
class ReviewRequest(BaseModel):
    review: str

    class Config:
        json_schema_extra = {
            "example": {
                "review": "This medication worked great for my condition. No side effects at all."
            }
        }


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_accuracy: float = None


# API Endpoints
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API and model are running"""
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model_loaded=model is not None,
        model_accuracy=metrics["accuracy"] if metrics else None
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    """Predict sentiment of a drug review"""
    if not model or not vectorizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty")

    # Transform text to features
    features = vectorizer.transform([request.review])

    # Get prediction and probabilities
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Map to label names
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = label_map[prediction]
    confidence = float(max(probabilities))

    return PredictionResponse(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        probabilities={
            "negative": round(float(probabilities[0]), 4),
            "neutral": round(float(probabilities[1]), 4),
            "positive": round(float(probabilities[2]), 4)
        }
    )


@app.post("/predict/batch")
def predict_batch(reviews: list[ReviewRequest]):
    """Predict sentiment for multiple reviews at once"""
    if not model or not vectorizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(reviews) > 100:
        raise HTTPException(status_code=400, detail="Max 100 reviews per batch")

    texts = [r.review for r in reviews]
    features = vectorizer.transform(texts)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    results = []
    for i, text in enumerate(texts):
        results.append({
            "review": text[:100] + "..." if len(text) > 100 else text,
            "sentiment": label_map[predictions[i]],
            "confidence": round(float(max(probabilities[i])), 4)
        })

    return {"predictions": results, "total": len(results)}
