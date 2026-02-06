"""
BERT-based Sentiment Model Training
Fine-tunes DistilBERT for drug review sentiment analysis
Uses DistilBERT (smaller, faster) instead of full BERT for practical deployment
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score
)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)


class ReviewDataset(Dataset):
    """Custom PyTorch Dataset for drug reviews"""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }


def load_data(sample_size=10000):
    """
    Load cleaned data with sampling for faster training
    Full dataset is 160K+ rows - we sample for practical training time
    """
    train_df = pd.read_csv("data/processed/train_clean.csv")
    test_df = pd.read_csv("data/processed/test_clean.csv")

    # Sample for practical training (BERT is much slower than sklearn)
    if len(train_df) > sample_size:
        train_df = train_df.groupby("sentiment", group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), sample_size // 3),
                random_state=42
            )
        )
    if len(test_df) > sample_size // 3:
        test_df = test_df.groupby("sentiment", group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), sample_size // 9),
                random_state=42
            )
        )

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train distribution:\n{train_df['sentiment'].value_counts()}")

    return train_df, test_df


def train_bert_model(train_df, test_df, epochs=3, batch_size=16, lr=2e-5):
    """Fine-tune DistilBERT for sentiment classification"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load tokenizer and model
    print("Loading DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3  # negative, neutral, positive
    )
    model.to(device)

    # Create datasets
    train_dataset = ReviewDataset(
        train_df["clean_review"].values,
        train_df["sentiment"].values,
        tokenizer
    )
    test_dataset = ReviewDataset(
        test_df["clean_review"].values,
        test_df["sentiment"].values,
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(
            f"  Epoch {epoch+1} complete | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Train Accuracy: {epoch_acc:.4f}"
        )

    return model, tokenizer, test_loader


def evaluate_bert(model, test_loader, device=None):
    """Evaluate BERT model on test set"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predictions = torch.argmax(outputs.logits, dim=1).cpu()
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())

    y_pred = np.array(all_predictions)
    y_test = np.array(all_labels)

    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_per_class = f1_score(y_test, y_pred, average=None)

    print("\n" + "=" * 50)
    print("BERT MODEL EVALUATION")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")

    target_names = ["negative", "neutral", "positive"]
    print(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")

    metrics = {
        "model_type": "distilbert",
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_per_class": {
            "negative": round(float(f1_per_class[0]), 4),
            "neutral": round(float(f1_per_class[1]), 4),
            "positive": round(float(f1_per_class[2]), 4)
        }
    }
    return metrics


def save_bert_model(model, tokenizer, metrics, output_dir="models/bert"):
    """Save BERT model and metrics"""
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nBERT model saved to {output_dir}/")

    # Compare with baseline
    baseline_path = "models/metrics.json"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        print("\n" + "=" * 50)
        print("MODEL COMPARISON")
        print("=" * 50)
        print(f"{'Metric':<20} {'Logistic Reg':<15} {'DistilBERT':<15}")
        print(f"{'Accuracy':<20} {baseline['accuracy']:<15} {metrics['accuracy']:<15}")
        print(f"{'F1 Weighted':<20} {baseline['f1_weighted']:<15} {metrics['f1_weighted']:<15}")


if __name__ == "__main__":
    print("=" * 50)
    print("BERT SENTIMENT MODEL TRAINING")
    print("=" * 50)

    # Load data (sampled for practical training)
    train_df, test_df = load_data(sample_size=10000)

    # Train
    model, tokenizer, test_loader = train_bert_model(
        train_df, test_df,
        epochs=3,
        batch_size=16
    )

    # Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_bert(model, test_loader, device)

    # Save
    save_bert_model(model, tokenizer, metrics)

    print("\n✅ BERT training complete!")