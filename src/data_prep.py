"""
Data Preparation Module for Drug Review Sentiment Analysis
Loads raw TSV data, cleans it, and creates sentiment labels
"""

import pandas as pd
import os
import re


def load_raw_data(data_dir="data/raw"):
    """Load train and test TSV files"""
    train_df = pd.read_csv(
        os.path.join(data_dir, "drugsComTrain_raw.tsv"),
        sep="\t",
        encoding="utf-8"
    )
    test_df = pd.read_csv(
        os.path.join(data_dir, "drugsComTest_raw.tsv"),
        sep="\t",
        encoding="utf-8"
    )
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    return train_df, test_df


def clean_text(text):
    """Clean review text"""
    if not isinstance(text, str):
        return ""
    # Remove HTML entities
    text = re.sub(r"&#\d+;", "'", text)
    text = re.sub(r"&\w+;", " ", text)
    # Remove extra quotes
    text = text.strip('"').strip("'")
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_sentiment_label(rating):
    """
    Convert 1-10 rating to sentiment:
    1-4  = negative (0)
    5-6  = neutral  (1)
    7-10 = positive (2)
    """
    if rating <= 4:
        return 0
    elif rating <= 6:
        return 1
    else:
        return 2


def get_sentiment_name(label):
    """Map label to readable name"""
    mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return mapping.get(label, "unknown")


def prepare_data(train_df, test_df):
    """Full data cleaning pipeline"""

    cleaned = []
    for df in [train_df, test_df]:
        # Drop rows with missing reviews or ratings
        df = df.dropna(subset=["review", "rating"]).copy()

        # Clean the review text
        df["clean_review"] = df["review"].apply(clean_text)

        # Remove empty reviews after cleaning
        df = df[df["clean_review"].str.len() > 10].copy()

        # Create sentiment labels
        df["sentiment"] = df["rating"].apply(create_sentiment_label)
        df["sentiment_name"] = df["sentiment"].apply(get_sentiment_name)

        cleaned.append(df)

    train_df, test_df = cleaned

    # Keep only needed columns
    columns_to_keep = [
        "drugName", "condition", "clean_review",
        "rating", "sentiment", "sentiment_name"
    ]
    train_clean = train_df[columns_to_keep].copy()
    test_clean = test_df[columns_to_keep].copy()

    print(f"\nCleaned Train shape: {train_clean.shape}")
    print(f"Cleaned Test shape: {test_clean.shape}")
    print(f"\nSentiment distribution (Train):")
    print(train_clean["sentiment_name"].value_counts())

    return train_clean, test_clean


def save_clean_data(train_clean, test_clean, output_dir="data/processed"):
    """Save cleaned data"""
    os.makedirs(output_dir, exist_ok=True)
    train_clean.to_csv(
        os.path.join(output_dir, "train_clean.csv"), index=False
    )
    test_clean.to_csv(
        os.path.join(output_dir, "test_clean.csv"), index=False
    )
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    print("=" * 50)
    print("DRUG REVIEW DATA PREPARATION")
    print("=" * 50)

    train_df, test_df = load_raw_data()
    train_clean, test_clean = prepare_data(train_df, test_df)
    save_clean_data(train_clean, test_clean)

    print("\n✅ Data preparation complete!")
