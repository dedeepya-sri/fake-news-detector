import streamlit as st
import tensorflow as tf
import pickle

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Load model safely
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/fake_news_model.keras", compile=False)

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    return pickle.load(open("model/tokenizer.pkl", "rb"))

model = load_model()
tokenizer = load_tokenizer()

# UI Design
st.title("📰 Fake News Detector")
st.markdown("### Check whether a news article is **Real or Fake**")

# Input box (PERSISTS TEXT)
user_input = st.text_area("Enter News Article", height=200)

# Button
if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Preprocess input
        seq = tokenizer.texts_to_sequences([user_input])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=200)

        # Prediction
        prediction = model.predict(padded)[0][0]

        # Output
        if prediction > 0.5:
            st.success("✅ This looks like REAL news")
        else:
            st.error("⚠️ This looks like FAKE news")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")