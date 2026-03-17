from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = tf.keras.models.load_model("model/fake_news_model.keras")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def extract_text_from_url(url):

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        article = ""

        for p in soup.select("article p"):
            article += p.get_text()

        if article == "":
            for p in soup.find_all("p"):
                article += p.get_text()

        return article

    except:
        return ""


def predict_news(text):

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=500)

    pred = model.predict(padded)[0][0]

    if pred > 0.5:
        return "Real News"
    else:
        return "Fake News"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    news_text = request.form.get("news", "")
    url = request.form.get("url", "")

    if url:
        news_text = extract_text_from_url(url)

    if news_text == "":
        result = "Could not extract article text."
    else:
        result = predict_news(news_text)

    return render_template(
        "index.html",
        prediction=result,
        news=news_text,
        url=url
    )


if __name__ == "__main__":
    app.run(debug=True)