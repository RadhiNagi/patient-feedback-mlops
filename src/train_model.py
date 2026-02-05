"""
Model Training Module
Trains a sentiment analysis model on cleaned drug review data
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight


def load_clean_data(data_dir="data/processed"):
    """Load the cleaned training and test data"""
    train_df = pd.read_csv(os.path.join(data_dir, "train_clean.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_clean.csv"))

    print(f"Train: {train_df.shape[0]} reviews")
    print(f"Test: {test_df.shape[0]} reviews")
    return train_df, test_df


def create_features(train_df, test_df, max_features=50000):
    """
    Convert text reviews into numbers using TF-IDF
    (Term Frequency - Inverse Document Frequency)
    """
    print(f"\nCreating TF-IDF features (max {max_features} words)...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),     # use single words AND two-word phrases
        min_df=5,               # word must appear in at least 5 reviews
        max_df=0.95,            # ignore words in 95%+ of reviews
        strip_accents="unicode",
        sublinear_tf=True       # apply log scaling
    )

    X_train = vectorizer.fit_transform(train_df["clean_review"])
    X_test = vectorizer.transform(test_df["clean_review"])

    y_train = train_df["sentiment"].values
    y_test = test_df["sentiment"].values

    print(f"Feature matrix shape: {X_train.shape}")
    print(f"(means {X_train.shape[0]} reviews, {X_train.shape[1]} word features)")

    return X_train, X_test, y_train, y_test, vectorizer


def train_model(X_train, y_train):
    """
    Train Logistic Regression with class weights
    to handle imbalanced data (more positive than negative reviews)
    """
    print("\nTraining model...")
    print("Using 'balanced' class weights to handle imbalanced data")

    model = LogisticRegression(
        class_weight="balanced",   # automatically adjusts for imbalance
        max_iter=1000,
        C=1.0,
        random_state=42,
        solver="lbfgs",
        multi_class="multinomial"
    )

    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    print("\n" + "=" * 50)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 50)

    y_pred = model.predict(X_test)

    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_per_class = f1_score(y_test, y_pred, average=None)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")

    # Detailed report
    target_names = ["negative", "neutral", "positive"]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(f"\nClassification Report:\n{report}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Neg   Neu   Pos")
    for i, name in enumerate(target_names):
        print(f"  Actual {name:>8}: {cm[i]}")

    # Save metrics as dictionary
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_per_class": {
            "negative": round(f1_per_class[0], 4),
            "neutral": round(f1_per_class[1], 4),
            "positive": round(f1_per_class[2], 4)
        }
    }
    return metrics


def save_artifacts(model, vectorizer, metrics, output_dir="models"):
    """Save model, vectorizer, and metrics"""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "sentiment_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path}")

    # Save vectorizer
    vec_path = os.path.join(output_dir, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vec_path)
    print(f"Vectorizer saved: {vec_path}")

    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("SENTIMENT MODEL TRAINING")
    print("=" * 50)

    # Step 1: Load data
    train_df, test_df = load_clean_data()

    # Step 2: Create features
    X_train, X_test, y_train, y_test, vectorizer = create_features(
        train_df, test_df
    )

    # Step 3: Train model
    model = train_model(X_train, y_train)

    # Step 4: Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Step 5: Save everything
    save_artifacts(model, vectorizer, metrics)

    print("\n✅ Training pipeline complete!")