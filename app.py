"""Smart Sentiment Analyzer - Modern AI-Powered Review Intelligence Dashboard."""
import json
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

from sentiment.model import (
    MODEL_PATH,
    get_matched_features,
    load_model,
    predict_proba,
    top_words,
)

# Page configuration
st.set_page_config(
    page_title="Smart Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Glassmorphism, Google Font, Modern Cards & Chips)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Header */
    .hero-container {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: rgba(99, 102, 241, 0.15);
        color: #6366F1;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 9999px;
        margin-bottom: 0.75rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E293B 0%, #4F46E5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }

    @media (prefers-color-scheme: dark) {
        .hero-title {
            background: linear-gradient(135deg, #F8FAFC 0%, #A5B4FC 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: #64748B;
        margin: 0;
        line-height: 1.5;
    }

    /* Sentiment Result Card */
    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1.25rem 0;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        transition: all 0.2s ease-in-out;
    }

    .card-pos {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        box-shadow: 0 8px 24px -6px rgba(16, 185, 129, 0.15);
    }

    .card-neu {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.35);
        box-shadow: 0 8px 24px -6px rgba(245, 158, 11, 0.15);
    }

    .card-neg {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.1) 0%, rgba(225, 29, 72, 0.05) 100%);
        border: 1px solid rgba(244, 63, 94, 0.35);
        box-shadow: 0 8px 24px -6px rgba(244, 63, 94, 0.15);
    }

    .sentiment-emoji {
        font-size: 3rem;
        line-height: 1;
    }

    .sentiment-label {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }

    .sentiment-conf {
        font-size: 0.95rem;
        font-weight: 600;
        color: #64748B;
        margin-top: 2px;
    }

    /* Keyword Tags */
    .keyword-tag {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px 4px;
        font-size: 0.82rem;
        font-weight: 600;
        border-radius: 8px;
        background: rgba(99, 102, 241, 0.12);
        color: #4F46E5;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    /* Quick Preset Chips */
    .chip-btn {
        background: transparent;
        border-radius: 8px;
        font-size: 0.82rem;
    }

    /* Custom Metric Box */
    .kpi-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }
    .kpi-val {
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 4px;
    }

    /* Sidebar Model Card */
    .sidebar-card {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.15);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div class="hero-container">
        <span class="badge-pill">⚡ NLP Intelligence Engine v2.0</span>
        <h1 class="hero-title">Smart Sentiment Analyzer</h1>
        <p class="hero-subtitle">
            Automated sentiment classification for product reviews and customer feedback using 
            n-gram TF-IDF representations & regularized Logistic Regression.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Verify Model Exists
if not MODEL_PATH.exists():
    st.error(
        "⚠️ **No trained model found!**\n\n"
        "Run the following commands in your terminal to initialize and train the model:\n"
        "```powershell\n"
        "python -m sentiment.sample_data\n"
        "python -m sentiment.train\n"
        "```"
    )
    st.stop()


@st.cache_resource
def get_model():
    return load_model()


model = get_model()
EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
COLOR_MAP = {
    "Positive": "#10B981",  # Emerald
    "Neutral": "#F59E0B",   # Amber
    "Negative": "#F43F5E",  # Rose Red
}

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.subheader("⚙️ Model Details")

    metrics_path = Path(__file__).resolve().parent / "models" / "metrics.json"
    if metrics_path.exists():
        try:
            m_data = json.loads(metrics_path.read_text())
            st.markdown(
                f"""
                <div class="sidebar-card">
                    <p style="margin:0 0 6px 0; font-size:0.85rem; color:#64748B;">Algorithm</p>
                    <p style="margin:0 0 12px 0; font-weight:700;">{m_data.get('model', 'Logistic Regression')}</p>
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <span style="font-size:0.85rem; color:#64748B;">Test Accuracy</span>
                        <strong style="color:#10B981;">{m_data.get('test_accuracy', 0):.1%}</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="font-size:0.85rem; color:#64748B;">Macro F1 Score</span>
                        <strong style="color:#6366F1;">{m_data.get('test_macro_f1', 0):.3f}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    st.markdown("---")
    st.subheader("📊 Model Artifacts")
    cm_path = Path(__file__).resolve().parent / "models" / "confusion_matrix.png"
    if cm_path.exists():
        with st.expander("Show Confusion Matrix", expanded=False):
            st.image(str(cm_path), caption="Evaluation Confusion Matrix")

    st.markdown("---")
    st.caption("Developed with scikit-learn, TF-IDF & Streamlit.")


# ----------------- MAIN TABS -----------------
tab1, tab2, tab3 = st.tabs(["🎯 Single Review Analysis", "📁 Batch Intelligence", "🔍 Key Drivers & Lexicon"])

# ================= TAB 1: SINGLE REVIEW =================
with tab1:
    st.write("#### Enter or choose a sample review to analyze:")

    # Quick Preset Buttons
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    if c_btn1.button("✨ Sample Positive", use_container_width=True):
        st.session_state["review_input"] = "The battery life is great, super fast performance and I absolutely love it!"
    if c_btn2.button("⚖️ Sample Neutral", use_container_width=True):
        st.session_state["review_input"] = "The product is okay, average build quality and does the job as expected."
    if c_btn3.button("💥 Sample Negative", use_container_width=True):
        st.session_state["review_input"] = "Horrible experience, completely broke in two days and total waste of money."

    default_text = st.session_state.get(
        "review_input", "The battery life is great, super fast performance and I absolutely love it!"
    )
    user_review = st.text_area("Review text", default_text, height=110, label_visibility="collapsed")

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    if (analyze_clicked or "has_run" not in st.session_state) and user_review.strip():
        st.session_state["has_run"] = True
        probs_df = predict_proba(model, [user_review])
        probs = probs_df.iloc[0]
        label = probs.idxmax()
        confidence = probs[label]
        matched_words = get_matched_features(model, user_review)

        # Style card depending on class
        card_class = "card-pos" if label == "Positive" else "card-neg" if label == "Negative" else "card-neu"
        label_color = COLOR_MAP[label]

        # Hero Verdict Card
        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="sentiment-emoji">{EMOJI[label]}</div>
                <div>
                    <h2 class="sentiment-label" style="color: {label_color};">{label}</h2>
                    <div class="sentiment-conf">Confidence Score: <strong>{confidence:.1%}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Columns for Visuals + Keyword insights
        c_chart, c_keywords = st.columns([3, 2])

        with c_chart:
            st.write("##### **Probability Distribution**")
            chart_data = pd.DataFrame({
                "Sentiment": list(probs.index),
                "Probability": list(probs.values),
                "Color": [COLOR_MAP[c] for c in probs.index]
            })

            bar_chart = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, height=26)
                .encode(
                    x=alt.X("Probability:Q", axis=alt.Axis(format="%", title=None), scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Sentiment:N", sort=["Positive", "Neutral", "Negative"], title=None, axis=alt.Axis(labelFontSize=13)),
                    color=alt.Color("Sentiment:N", scale=alt.Scale(domain=["Positive", "Neutral", "Negative"], range=["#10B981", "#F59E0B", "#F43F5E"]), legend=None),
                    tooltip=[alt.Tooltip("Sentiment:N"), alt.Tooltip("Probability:Q", format=".1%")],
                )
                .properties(height=160)
            )

            # Add percentage text inside/beside bars
            text_chart = bar_chart.mark_text(
                align="left",
                baseline="middle",
                dx=6,
                fontWeight="bold",
                fontSize=12
            ).encode(
                text=alt.Text("Probability:Q", format=".1%")
            )

            st.altair_chart(bar_chart + text_chart, use_container_width=True)

        with c_keywords:
            st.write("##### **Extracted Key Phrases**")
            if matched_words:
                tags_html = "".join([f'<span class="keyword-tag">🏷️ {w}</span>' for w in matched_words])
                st.markdown(f'<div style="margin-top: 8px;">{tags_html}</div>', unsafe_allow_html=True)
                st.caption("N-gram features that directly influenced the model's decision.")
            else:
                st.warning(
                    "⚠️ **Zero vocabulary matches found.**\n\n"
                    "None of the words appeared in the model's training dictionary. "
                    "The prediction reflects the default baseline distribution."
                )

# ================= TAB 2: BATCH INTELLIGENCE =================
with tab2:
    st.write("#### Analyze Bulk Customer Feedback")
    st.caption("Upload a `.csv` with a `review` / `text` column, or paste one review per line.")

    col_up, col_paste = st.columns(2)
    with col_up:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")
    with col_paste:
        pasted_text = st.text_area("...or paste reviews (one per line)", height=115, placeholder="Paste one review per line here...")

    batch_reviews = None
    if uploaded_file is not None:
        try:
            df_in = pd.read_csv(uploaded_file)
            col = next((c for c in ("review", "text", "reviewText", "Review") if c in df_in.columns), None)
            if col is None:
                st.error("CSV must contain a 'review', 'text', or 'Review' column.")
            else:
                batch_reviews = df_in[col].dropna().astype(str)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif pasted_text.strip():
        batch_reviews = pd.Series([line.strip() for line in pasted_text.splitlines() if line.strip()])

    if batch_reviews is not None and len(batch_reviews) > 0:
        with st.spinner("Processing batch sentiment..."):
            preds = model.predict(list(batch_reviews))
            probs_matrix = predict_proba(model, batch_reviews)
            max_conf = probs_matrix.max(axis=1)

            res_df = pd.DataFrame({
                "Review": batch_reviews.values,
                "Sentiment": preds,
                "Confidence": max_conf.values
            })

            counts = res_df["Sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).astype(int)
            total = len(res_df)

        st.markdown("---")
        st.write("### 📈 Batch Overview")

        # KPI Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="kpi-box"><div class="kpi-title">Total Reviews</div><div class="kpi-val" style="color:#6366F1;">{total}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="kpi-box"><div class="kpi-title">Positive</div><div class="kpi-val" style="color:#10B981;">{counts["Positive"]} <span style="font-size:0.9rem; font-weight:600;">({counts["Positive"]/total:.0%})</span></div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="kpi-box"><div class="kpi-title">Neutral</div><div class="kpi-val" style="color:#F59E0B;">{counts["Neutral"]} <span style="font-size:0.9rem; font-weight:600;">({counts["Neutral"]/total:.0%})</span></div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="kpi-box"><div class="kpi-title">Negative</div><div class="kpi-val" style="color:#F43F5E;">{counts["Negative"]} <span style="font-size:0.9rem; font-weight:600;">({counts["Negative"]/total:.0%})</span></div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Donut Chart for Sentiment Breakdown
        donut_data = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [counts["Positive"], counts["Neutral"], counts["Negative"]]
        })

        donut_chart = (
            alt.Chart(donut_data)
            .mark_arc(innerRadius=65, stroke="#fff", strokeWidth=2)
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color(
                    "Sentiment:N",
                    scale=alt.Scale(domain=["Positive", "Neutral", "Negative"], range=["#10B981", "#F59E0B", "#F43F5E"]),
                    legend=alt.Legend(orient="bottom", title=None)
                ),
                tooltip=["Sentiment:N", "Count:Q"]
            )
            .properties(height=260)
        )

        col_donut, col_table = st.columns([1, 2])
        with col_donut:
            st.altair_chart(donut_chart, use_container_width=True)

        with col_table:
            filter_choice = st.selectbox("Filter Sentiment:", ["All", "Positive", "Neutral", "Negative"])
            display_df = res_df if filter_choice == "All" else res_df[res_df["Sentiment"] == filter_choice]

            # Format confidence
            formatted_df = display_df.copy()
            formatted_df["Confidence"] = formatted_df["Confidence"].apply(lambda x: f"{x:.1%}")

            st.dataframe(formatted_df, use_container_width=True, height=220)

            csv_data = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Results as CSV",
                data=csv_data,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )

# ================= TAB 3: KEY DRIVERS & LEXICON =================
with tab3:
    st.write("#### 🔍 Class-Indicative Words & N-Grams")
    st.caption("Inspect the top features that the model associates most strongly with each sentiment category.")

    n_words = st.slider("Number of top features per class:", min_value=5, max_value=30, value=15)
    words_df = top_words(model, n_words)

    col_pos, col_neu, col_neg = st.columns(3)

    with col_pos:
        st.markdown(
            f"""
            <div style="background:rgba(16, 185, 129, 0.08); padding:1rem; border-radius:12px; border:1px solid rgba(16, 185, 129, 0.25);">
                <h4 style="color:#10B981; margin:0 0 10px 0;">😊 Positive Drivers</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        if "Positive" in words_df.columns:
            for idx, w in enumerate(words_df["Positive"], 1):
                st.markdown(f"**{idx}.** `{w}`")

    with col_neu:
        st.markdown(
            f"""
            <div style="background:rgba(245, 158, 11, 0.08); padding:1rem; border-radius:12px; border:1px solid rgba(245, 158, 11, 0.25);">
                <h4 style="color:#F59E0B; margin:0 0 10px 0;">😐 Neutral Drivers</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        if "Neutral" in words_df.columns:
            for idx, w in enumerate(words_df["Neutral"], 1):
                st.markdown(f"**{idx}.** `{w}`")

    with col_neg:
        st.markdown(
            f"""
            <div style="background:rgba(244, 63, 94, 0.08); padding:1rem; border-radius:12px; border:1px solid rgba(244, 63, 94, 0.25);">
                <h4 style="color:#F43F5E; margin:0 0 10px 0;">😞 Negative Drivers</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        if "Negative" in words_df.columns:
            for idx, w in enumerate(words_df["Negative"], 1):
                st.markdown(f"**{idx}.** `{w}`")
