from flask import Flask, render_template, request
import joblib
import spacy
from roberta_model import *

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    roberta_result = None
    text = ""
    if request.method == 'POST':
        text = request.form['text']
        roberta_result = analyze_sentiment_roberta(text)
    return render_template('index.html', text=text, roberta_result=roberta_result)

if __name__ == '__main__':
    app.run(debug=True)