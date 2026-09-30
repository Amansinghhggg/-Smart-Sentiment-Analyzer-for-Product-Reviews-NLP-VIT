import pytest
from sentiment.preprocess import clean_text
from sentiment.sample_data import make_dataset
from sentiment.train import rating_to_label, build_pipeline, split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def test_clean_text_removes_html_urls_and_keeps_negation():
    out = clean_text("<b>NOT</b> good!! See https://x.com/abc")
    assert "not" in out.split() and "good" in out.split()
    assert "http" not in out and "<" not in out


def test_contraction_becomes_negation():
    assert "not" in clean_text("It doesn't work").split()


@pytest.mark.parametrize("rating,label", [(1, "Negative"), (2, "Negative"), (3, "Neutral"), (4, "Positive"), (5, "Positive")])
def test_rating_mapping(rating, label):
    assert rating_to_label(rating) == label


@pytest.fixture(scope="module")
def trained():
    df = make_dataset(150)
    df["label"] = df["rating"].map(rating_to_label)
    X_tr, X_dev, X_te, y_tr, y_dev, y_te = split(df)
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr), X_te, y_te


def test_model_accuracy_on_test_split(trained):
    model, X_te, y_te = trained
    assert accuracy_score(y_te, model.predict(X_te)) > 0.85


@pytest.mark.parametrize("text,expected", [
    ("Excellent phone, I absolutely love it, highly recommend", "Positive"),
    ("Terrible product, broke in two days, total waste of money", "Negative"),
])
def test_obvious_reviews(trained, text, expected):
    assert trained[0].predict([text])[0] == expected
