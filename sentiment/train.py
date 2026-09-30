"""Train + evaluate classifiers. Usage: python -m sentiment.train [--data data/sample_reviews.csv]"""
import argparse, json
from pathlib import Path
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
from .preprocess import clean_text

LABELS = ["Negative", "Neutral", "Positive"]
ROOT = Path(__file__).resolve().parent.parent


def rating_to_label(r) -> str:
    r = float(r)
    return "Negative" if r <= 2 else "Neutral" if r < 4 else "Positive"


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = next(c for c in ("review", "text", "reviewText", "Review") if c in df.columns)
    if "label" in df.columns:
        labels = df["label"].astype(str).str.capitalize()
    else:
        rc = next(c for c in ("rating", "Rating", "overall", "stars") if c in df.columns)
        labels = df[rc].map(rating_to_label)
    out = pd.DataFrame({"review": df[text_col], "label": labels}).dropna()
    return out[out["label"].isin(LABELS)].reset_index(drop=True)


def build_pipeline(clf) -> Pipeline:
    tfidf = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def split(df, seed=42):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(df["review"], df["label"], test_size=0.30, stratify=df["label"], random_state=seed)
    X_dev, X_te, y_dev, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return X_tr, X_dev, X_te, y_tr, y_dev, y_te


def main(data_path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    print(f"Loaded {len(df)} reviews:\n{df['label'].value_counts().to_string()}\n")
    X_tr, X_dev, X_te, y_tr, y_dev, y_te = split(df)
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "MultinomialNB": MultinomialNB(),
    }
    best_name, best_model, best_f1 = None, None, -1
    for name, clf in candidates.items():
        pipe = build_pipeline(clf).fit(X_tr, y_tr)
        f1 = f1_score(y_dev, pipe.predict(X_dev), average="macro")
        print(f"[dev] {name}: macro-F1 = {f1:.3f}")
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, pipe, f1
    pred = best_model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print(f"\nBest model: {best_name}\n[test] accuracy = {acc:.3f}\n")
    print(classification_report(y_te, pred, labels=LABELS, digits=3))
    cm = confusion_matrix(y_te, pred, labels=LABELS)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues")
    plt.title(f"Confusion matrix - {best_name}"); plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=120); plt.close()
    errors = pd.DataFrame({"review": X_te, "true": y_te, "pred": pred}).query("true != pred")
    errors.head(20).to_csv(out_dir / "error_cases.csv", index=False)
    joblib.dump(best_model, out_dir / "sentiment_model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"model": best_name, "dev_macro_f1": best_f1, "test_accuracy": acc,
         "test_macro_f1": f1_score(y_te, pred, average="macro")}, indent=2))
    print(f"Saved model + reports to {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "sample_reviews.csv"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    a = ap.parse_args()
    main(a.data, a.out)
