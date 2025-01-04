from flask import Flask, render_template, request
import joblib
import spacy
from util import analyze_sentiment
from data_processing import load_emoji_sentiment_data
from roberta_model import *

app = Flask(__name__)

model = joblib.load("models/trained_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
nlp = spacy.load('en_core_web_sm')
emoji_sentiment_dict = load_emoji_sentiment_data("data/Emoji_Sentiment_Data.csv")


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    text = None
    roberta_result = None
    if request.method == 'POST':
        text = request.form['text']
        result = analyze_sentiment(text, nlp, model, vectorizer, emoji_sentiment_dict)
        roberta_result = analyze_sentiment_roberta(text)
    return render_template('index.html', result=result, text=text, roberta_result=roberta_result)

if __name__ == '__main__':
    app.run(debug=True)