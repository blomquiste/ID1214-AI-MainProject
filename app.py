from flask import Flask, render_template, request
import joblib
import spacy
from roberta_model_finetuned import *

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    roberta_result = None
    text = ""
    if request.method == 'POST':
        text = request.form['text']
        roberta_result = combined_analysis(text)
    return render_template('index.html', text=text, roberta_result=roberta_result)

if __name__ == '__main__':
    app.run(debug=True)