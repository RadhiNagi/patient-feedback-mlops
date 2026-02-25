# Interview Prep Guide (Patient Feedback MLOps Project)

This guide is designed for a fresher-friendly explanation so you can confidently explain your project in interviews.

## 1) 60-second project intro (say this first)

> "This is an end-to-end MLOps project for sentiment analysis on patient drug reviews. We clean raw TSV data, train a TF-IDF + Logistic Regression model, evaluate with accuracy/F1, save model artifacts, expose predictions through a FastAPI service, and automate tests/build quality checks using GitHub Actions CI/CD."

---

## 2) Project layers (industry way)

Think in **6 layers**:

1. **Data Layer**
   - Raw files in `data/raw/*.tsv`
   - Processed files in `data/processed/*.csv`

2. **Feature Engineering Layer**
   - Text cleaning + sentiment label mapping in `src/data_prep.py`
   - TF-IDF vectorization in `src/train_model.py`

3. **Model Layer**
   - Multi-class Logistic Regression (`negative`, `neutral`, `positive`)
   - Evaluation and metrics generation in `src/train_model.py`

4. **Model Registry / Artifact Layer**
   - Saved artifacts in `models/`
   - Version metadata in `src/model_registry.py`

5. **Serving Layer (API)**
   - FastAPI app with `/health`, `/predict`, `/predict/batch` in `src/predict_api.py`

6. **DevOps/MLOps Layer**
   - Docker + Docker Compose
   - CI/CD automation in `.github/workflows/ci-cd.yml`
   - Automated tests in `tests/`

---

## 3) Code flow (end-to-end)

### Step A: Data Preparation
- File: `src/data_prep.py`
- Key flow:
  1. Load train and test TSV.
  2. Clean text (remove entities/noise/extra spaces).
  3. Convert rating → sentiment class.
  4. Save cleaned CSV.

Command:
```bash
python -m src.data_prep
```

### Step B: Model Training
- File: `src/train_model.py`
- Key flow:
  1. Load cleaned CSV.
  2. Create TF-IDF features.
  3. Train Logistic Regression.
  4. Evaluate on test set (accuracy/F1/confusion matrix).
  5. Save model + vectorizer + metrics.
  6. Register model version.

Command:
```bash
python -m src.train_model
```

### Step C: Model Serving (API)
- File: `src/predict_api.py`
- Key flow:
  1. Load saved model/vectorizer at startup.
  2. `/predict` for single text.
  3. `/predict/batch` for multiple texts.
  4. `/health` for monitoring model status.

Command:
```bash
uvicorn src.predict_api:app --host 0.0.0.0 --port 8000
```

### Step D: Automated Validation
- File: `tests/`
- Commands:
```bash
pytest
```

### Step E: Containerized Run
```bash
docker compose run data-prep
docker compose run train
docker compose run test
docker compose up api
```

---

## 4) Interview concepts to explain clearly

### Why TF-IDF + Logistic Regression?
- Fast baseline, interpretable, production-friendly, lower compute than deep models.

### Why weighted F1?
- Dataset is class-imbalanced; weighted F1 reflects performance across all classes better than plain accuracy.

### Why CI/CD for ML?
- Prevents broken code/model from reaching production.
- Enforces tests + quality gate (accuracy/F1 thresholds).

### Why `/health` endpoint?
- Ops teams can quickly verify API/model readiness for monitoring and alerts.

---

## 5) Git basics for fresher (must know)

## Core terms
- **Repository (repo):** your project folder tracked by Git.
- **Commit:** checkpoint/snapshot of code changes.
- **Branch:** separate line of development.
- **PR (Pull Request):** request to merge branch changes into main branch.
- **Code Review:** teammates review PR for quality, bugs, readability.

## Daily commands
```bash
git status
git pull origin main
git checkout -b feature/my-change
# edit files
git add .
git commit -m "feat: add batch input validation"
git push origin feature/my-change
```

Then open a **PR** on GitHub.

---

## 6) What interviewers expect from freshers

1. Can explain project flow in simple order.
2. Knows why each tech was used.
3. Knows basic Git + PR workflow.
4. Can discuss one bug and one fix.
5. Understands testing and deployment basics.
6. Is honest about limitations and improvement plan.

---

## 7) Bug + fix example from this project (great to discuss)

### Bug found
Batch API accepted empty requests and blank review entries, which can lead to low-quality predictions and poor API contract clarity.

### Fix implemented
Added validation in `/predict/batch` to:
- reject empty list input,
- reject blank review text and return invalid indexes.

Why this is industry-level:
- Better API contract,
- safer production behavior,
- easier client-side debugging,
- covered by automated tests.

---

## 8) Small speaking template for interviews

Use this structure:

1. **Problem**: Need scalable sentiment analysis on patient feedback.
2. **Data**: Drug reviews + numeric rating.
3. **ML approach**: Clean text → TF-IDF → Logistic Regression.
4. **Evaluation**: Accuracy + weighted F1 + class-wise analysis.
5. **Deployment**: FastAPI + Docker.
6. **MLOps**: CI/CD pipeline with tests and quality gates.
7. **My contribution**: data prep, model training, API, tests, bug fixes.
8. **Improvement ideas**: BERT model, monitoring dashboard, retraining pipeline.

---

## 9) Quick revision checklist (night before interview)

- [ ] Can explain all 6 layers without reading notes.
- [ ] Can describe 3 endpoints and sample request/response.
- [ ] Can explain metrics: accuracy vs weighted F1.
- [ ] Can explain PR and code review in 3 lines.
- [ ] Can explain one bug and one fix from your own code.
- [ ] Can run core commands once locally.

Good luck — focus on clarity and honesty. Interviewers reward structured thinking more than fancy words.
