import pandas as pd
import streamlit as st

from src.inference import predict_sentiment, clean_text

st.set_page_config(
    page_title="Amazon Echo Review Sentiment Analyzer",
    page_icon="🗣️",
    layout="centered",
)

st.title("🗣️ Amazon Echo Review Sentiment Analyzer")
st.write(
    "This app analyzes Amazon Echo Dot reviews and predicts whether the sentiment is "
    "**Negative**, **Neutral**, or **Positive** using a machine learning model."
)

st.markdown("---")

example_reviews = [
    "I love my Echo Dot, it works so well!",
    "This device is terrible, waste of money.",
    "It's okay but the speaker is weak.",
    "Awesome product! I use it every day.",
    "Not great speakers, I'm disappointed.",
]

with st.expander("Try an example review"):
    selected = st.selectbox("Select an example:", ["(none)"] + example_reviews)

if selected != "(none)":
    default_text = selected
else:
    default_text = ""

review_text = st.text_area(
    "✍️ Enter an Amazon Echo review:",
    value=default_text,
    height=150,
    placeholder="Type or paste a review here...",
)

if st.button("Analyze Now", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        label, confidence, probs = predict_sentiment(review_text)

        emoji_map = {"Negative": "😢", "Neutral": "😎", "Positive": "✅"}
        color_map = {"Negative": "red", "Neutral": "orange", "Positive": "green"}

        emoji = emoji_map[label]
        color = color_map[label]

        st.markdown("### Result")
        st.markdown(
            f"<h3 style='color:{color};'>"
            f"{emoji} Predicted Sentiment: {label} "
            f"({confidence:.2%} confidence)</h3>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Class Probabilities")
        prob_df = pd.DataFrame(
            {
                "Sentiment": ["Negative", "Neutral", "Positive"],
                "Probability": probs,
            }
        ).set_index("Sentiment")

        st.bar_chart(prob_df)

        with st.expander("See cleaned text (for NLP understanding)"):
            st.code(clean_text(review_text))

st.markdown("---")
st.caption("CWP Academy project.")
