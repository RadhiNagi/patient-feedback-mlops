# 🏥 Healthcare Drug Review Sentiment Analysis - MLOps Pipeline

![CI/CD](https://github.com/RadhiNagi/patient-feedback-mlops/actions/workflows/ci-cd.yml/badge.svg)

An end-to-end MLOps pipeline that analyzes patient drug reviews to predict sentiment (positive/negative/neutral), with automated CI/CD, model quality gating, and containerized deployment.

## 🏗️ Architecture
```
Raw Data (TSV) → Data Cleaning → Model Training → Testing → Docker → CI/CD → API
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 81.44% |
| Weighted F1 | 82.79% |
| Positive Precision | 95% |
| Negative Recall | 81% |

Trained on **160,939** drug reviews, tested on **53,631** reviews.

## 🛠️ Tech Stack

- **ML:** Python, Scikit-learn, TF-IDF, Logistic Regression
- **API:** FastAPI, Uvicorn
- **Containerization:** Docker, Docker Compose
- **CI/CD:** GitHub Actions (automated testing + model quality gate)
- **Testing:** Pytest (27 automated tests)

## 🚀 Quick Start

### Run with Docker Compose
```bash
# Prepare data
docker compose run data-prep

# Train model
docker compose run train

# Run tests
docker compose run test

# Start API
docker compose up api
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/predict` | POST | Predict single review sentiment |
| `/predict/batch` | POST | Predict batch (up to 100 reviews) |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This medication worked great!"}'
```

### Example Response
```json
{
  "sentiment": "positive",
  "confidence": 0.9154,
  "probabilities": {
    "negative": 0.0175,
    "neutral": 0.067,
    "positive": 0.9154
  }
}
```

## 🔄 CI/CD Pipeline

Every `git push` triggers:
1. **Test Job** - Installs dependencies, prepares data, trains model, runs 27 pytest tests
2. **Build Job** - Builds and verifies Docker image
3. **Model Quality Gate** - Blocks deployment if accuracy < 70% or F1 < 65%

## 📁 Project Structure
```
patient-feedback-mlops/
├── .github/workflows/ci-cd.yml   # CI/CD pipeline
├── data/raw/                      # Raw TSV data files
├── models/                        # Trained model artifacts
├── src/
│   ├── data_prep.py               # Data cleaning pipeline
│   ├── train_model.py             # Model training + evaluation
│   ├── predict_api.py             # FastAPI prediction service
│   └── model_registry.py          # Model version tracking
├── tests/
│   ├── test_data_prep.py          # Data cleaning tests
│   ├── test_model.py              # Model validation tests
│   └── test_api.py                # API endpoint tests
├── docker-compose.yml             # Multi-service orchestration
├── Dockerfile                     # Container definition
└── requirements.txt               # Python dependencies
```

## 📈 Future Enhancements

- AWS deployment with ECR + ECS
- Grafana monitoring dashboard
- Model A/B testing
- Deep learning model (BERT) comparison
- Automated retraining on new data
```
