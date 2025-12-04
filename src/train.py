import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ---------- Paths ----------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "amazon_alexa_reviews.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ---------- Helpers ----------

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, keep only letters + spaces, strip."""
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rating_to_label(rating: int) -> int:
    """
    Map star rating (1–5) to sentiment class:
    0 = Negative (1–2 stars)
    1 = Neutral  (3 stars)
    2 = Positive (4–5 stars)
    """
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


# ---------- Main Training Pipeline ----------

def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Drop missing reviews/ratings
    df = df.dropna(subset=["Review Text", "Rating"])

    # Ensure Rating is numeric (just in case)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating"])

    # Create multiclass sentiment label
    df["label"] = df["Rating"].apply(rating_to_label)

    print("Label distribution:")
    print(df["label"].value_counts())

    # Clean text
    print("Cleaning text...")
    df["clean_text"] = df["Review Text"].apply(clean_text)

    # Train / test split
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectorizer
    print("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Multinomial logistic regression
    print("Training logistic regression model...")
    model = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        class_weight="balanced",
    )
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)

    print("\n=== Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save artifacts
    model_path = MODELS_DIR / "amazon_echo_sentiment_model.joblib"
    vec_path = MODELS_DIR / "tf_vectorizer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(tfidf, vec_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved vectorizer to: {vec_path}")


if __name__ == "__main__":
    main()
