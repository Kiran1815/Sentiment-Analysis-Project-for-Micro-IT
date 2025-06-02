import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set up Streamlit page
st.set_page_config(page_title="SmartSent - 3-Class Sentiment", layout="centered")
st.title("🤖 SmartSent – Sentiment Analyzer")
st.markdown("Detects *Positive, **Negative, and **Neutral* sentiments using a RoBERTa transformer.")

# Load model and tokenizer
@st.cache_resource
def load_pipeline():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

analyzer = load_pipeline()

# User input
user_input = st.text_area("Enter text to analyze:", height=150)

# History setup
if "history" not in st.session_state:
    st.session_state.history = []

# Analyze text
if st.button("Analyze Sentiment") and user_input.strip():
    result = analyzer(user_input)[0]
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }

    label = label_map[result['label']]
    score = result['score']

    # Show result
    if label == "Positive":
        st.success(f"😊 Positive Sentiment\nConfidence: {score:.2f}")
    elif label == "Negative":
        st.error(f"😠 Negative Sentiment\nConfidence: {score:.2f}")
    else:
        st.info(f"😐 Neutral Sentiment\nConfidence: {score:.2f}")

    # Word cloud
    wordcloud = WordCloud(width=800, height=400).generate(user_input)
    st.subheader("☁ Word Cloud")
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # Save history
    st.session_state.history.append((user_input, label, score))

# Show past results
if st.session_state.history:
    st.subheader("📜 Analysis History")
    for i, (text, label, score) in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"{i}. *{label}* ({score:.2f}) — {text}")