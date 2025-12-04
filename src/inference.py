import re
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def clean_text(text: str) -> str:
    """Same cleaning used during training."""
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=1)
def load_artifacts():
    """Load trained model and vectorizer (cached)."""
    model_path = MODELS_DIR / "amazon_echo_sentiment_model.joblib"
    vec_path = MODELS_DIR / "tf_vectorizer.joblib"

    model = joblib.load(model_path)
    vec = joblib.load(vec_path)
    return model, vec


def predict_sentiment(review: str) -> Tuple[str, float, np.ndarray]:
    """
    Predict sentiment for a single review.

    Returns:
        label: "Negative" | "Neutral" | "Positive"
        confidence: probability of the predicted class
        probs: array of probabilities [p_neg, p_neu, p_pos]
    """
    model, vec = load_artifacts()

    clean = clean_text(review)
    features = vec.transform([clean])

    probs = model.predict_proba(features)[0]  # shape: (3,)
    pred_idx = int(np.argmax(probs))

    label = LABEL_MAP[pred_idx]
    confidence = float(probs[pred_idx])

    return label, confidence, probs
