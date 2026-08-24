import streamlit as st
import joblib
import re
import string


# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# Text cleaning
def clean_text(text):
    text = str(text)

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖"
)


# Title
st.title("🤖 AI Sentiment Analyzer")

st.write(
    "Enter a product review and let the AI predict its sentiment."
)


# User input
review = st.text_area(
    "Enter your review:",
    placeholder="Example: I really love this product!"
)


# Analyze
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        # Preprocess
        cleaned_review = clean_text(review)

        # TF-IDF transformation
        review_tfidf = tfidf.transform([cleaned_review])

        # Prediction
        prediction = model.predict(review_tfidf)[0]

        # Result
        if prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

        # Confidence
        probabilities = model.predict_proba(review_tfidf)[0]

        confidence = max(probabilities) * 100

        st.write(f"Confidence: {confidence:.2f}%")

        # -----------------------------------
# STREAMLIT USER INTERFACE
# -----------------------------------

st.title("🤖 AI Sentiment Analyzer")

st.write(
    "Enter a product review below and the AI model will predict "
    "whether the sentiment is Positive or Negative."
)

review = st.text_area(
    "Enter your review:",
    placeholder="Example: I really love this product. The quality is excellent!"
)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        # Clean the review
        clean_review = clean_text(review)

        # Apply the same preprocessing used during training
        processed_review = preprocess_text(clean_review)

        # Convert text into TF-IDF features
        review_vector = tfidf.transform([processed_review])

        # Predict sentiment
        prediction = model.predict(review_vector)[0]

        # Display result
        if prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")