"""Smart Sentiment Analyzer - Streamlit demo. Run: streamlit run app.py"""
import pandas as pd
import streamlit as st
from sentiment.model import MODEL_PATH, load_model, predict_proba, top_words, get_matched_features

st.set_page_config(page_title="Smart Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("💬 Smart Sentiment Analyzer")
st.caption("Classify product reviews as Positive, Negative or Neutral (TF-IDF + Logistic Regression / Naive Bayes).")

if not MODEL_PATH.exists():
    st.error("No trained model found. In a terminal run:  `python -m sentiment.sample_data` then `python -m sentiment.train`")
    st.stop()


@st.cache_resource
def get_model():
    return load_model()


model = get_model()
EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
tab1, tab2, tab3 = st.tabs(["Single review", "Batch analysis", "Top words"])

with tab1:
    st.subheader("Analyze Single Review")

    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Sample Positive"):
        st.session_state["review_input"] = "The battery life is great, super fast performance and I absolutely love it!"
    if col_b.button("Sample Neutral"):
        st.session_state["review_input"] = "The product is okay, average build quality and does the job as expected."
    if col_c.button("Sample Negative"):
        st.session_state["review_input"] = "Horrible experience, completely broke in two days and total waste of money."

    default_text = st.session_state.get("review_input", "The battery life is great, super fast performance and I absolutely love it!")
    text = st.text_area("Enter a product review", default_text, height=120)

    if st.button("Analyze", type="primary") and text.strip():
        matched = get_matched_features(model, text)
        probs = predict_proba(model, [text]).iloc[0]
        label = probs.idxmax()

        st.subheader(f"{EMOJI[label]} {label}  ({probs.max():.0%} confidence)")
        st.bar_chart(probs)

        if matched:
            st.markdown(f"**Identified Keywords:** `{', '.join(matched)}`")
        else:
            st.warning(
                "⚠️ **No recognizable sentiment keywords detected from the training vocabulary.**\n\n"
                "When all words are out-of-vocabulary, the TF-IDF vector is empty, so the model defaults "
                "to its prior baseline probability (~37% Neutral)."
            )

with tab2:
    st.write("Upload a CSV with a **review** column, or paste one review per line.")
    up = st.file_uploader("CSV file", type="csv")
    pasted = st.text_area("...or paste reviews (one per line)", height=120)
    reviews = None
    if up is not None:
        df = pd.read_csv(up)
        col = next((c for c in ("review", "text", "reviewText") if c in df.columns), None)
        if col is None:
            st.error("CSV needs a 'review' column.")
        else:
            reviews = df[col].astype(str)
    elif pasted.strip():
        reviews = pd.Series([l for l in pasted.splitlines() if l.strip()])
    if reviews is not None and len(reviews):
        res = pd.DataFrame({"review": reviews, "sentiment": model.predict(list(reviews))})
        counts = res["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).astype(int)
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", counts["Positive"]); c2.metric("Neutral", counts["Neutral"]); c3.metric("Negative", counts["Negative"])
        st.bar_chart(counts)
        overall = counts.idxmax()
        st.info(f"Overall sentiment: {EMOJI[overall]} **{overall}** ({counts.max() / counts.sum():.0%} of {counts.sum()} reviews)")
        st.dataframe(res, width="stretch")
        st.download_button("Download results (CSV)", res.to_csv(index=False), "sentiment_results.csv", "text/csv")

with tab3:
    n = st.slider("Words per class", 5, 25, 12)
    st.dataframe(top_words(model, n), width="stretch")
    st.caption("Most class-indicative words learned by the model.")
