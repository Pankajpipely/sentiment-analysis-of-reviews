import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.special import softmax
import torch

# --- Load RoBERTa model and tokenizer directly from Hugging Face ---
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# --- Initialize VADER ---
sia = SentimentIntensityAnalyzer()

# --- RoBERTa Sentiment Function ---
def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        "roberta_neg": f"{scores[0] * 100:.2f}%",
        "roberta_neu": f"{scores[1] * 100:.2f}%",
        "roberta_pos": f"{scores[2] * 100:.2f}%"
    }

# --- Convert VADER to Percentage ---
def vader_scores_percentage(text):
    vader_result = sia.polarity_scores(text)
    return {f"vader_{k}": f"{v * 100:.2f}%" for k, v in vader_result.items()}

# --- Streamlit App UI ---
st.title("Sentiment Analysis App")
st.write("Enter a review and get sentiment scores from **VADER** and **RoBERTa** (in %).")

review_text = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if review_text.strip():
        # VADER analysis
        vader_result_renamed = vader_scores_percentage(review_text)

        # RoBERTa analysis
        roberta_result = polarity_scores_roberta(review_text)

        # Combine results
        results = {**vader_result_renamed, **roberta_result}

        st.subheader("Sentiment Scores (%)")
        st.json(results)
    else:
        st.warning("Please enter some text before clicking **Analyze Sentiment**.")
