
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")

st.title("Sentiment Analyzer")
st.write("Enter a sentence below to analyze its sentiment.")

@st.cache_resource
def load_model():
    """Load the pretrained sentiment model once and reuse it."""
    return pipeline("sentiment-analysis")

try:
    classifier = load_model()
except Exception as e:
    st.error("The sentiment model could not be loaded.")
    st.write("Check your internet connection and try again.")
    st.stop()

text = st.text_area("Enter your text:", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            result = classifier(text.strip())[0]

            sentiment = result["label"]
            confidence = result["score"] * 100

            st.subheader("Analysis Result")
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error("Sorry, the text could not be analyzed.")
            st.write("Error details:", str(e))
