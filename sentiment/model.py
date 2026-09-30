"""Helpers used by the Streamlit app: load model, predict, inspect top words."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment_model.joblib"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_proba(model, texts) -> pd.DataFrame:
    return pd.DataFrame(model.predict_proba(list(texts)), columns=list(model.classes_))


def top_words(model, n=10) -> pd.DataFrame:
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    scores = clf.coef_ if hasattr(clf, "coef_") else clf.feature_log_prob_ - clf.feature_log_prob_.mean(axis=0)
    return pd.DataFrame({c: names[np.argsort(scores[i])[::-1][:n]] for i, c in enumerate(clf.classes_)})


def get_matched_features(model, text: str) -> list[str]:
    """Return recognized n-gram tokens from the model vocabulary found in text."""
    try:
        vec = model.named_steps["tfidf"]
        analyze = vec.build_analyzer()
        tokens = analyze(text)
        vocab = vec.vocabulary_
        matched = [t for t in tokens if t in vocab]
        # Return unique tokens preserving order
        return list(dict.fromkeys(matched))
    except Exception:
        return []
