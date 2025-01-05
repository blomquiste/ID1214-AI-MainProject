from flask import Flask, render_template, request
from roberta_model_finetuned import *
from user_input import analyze

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    svm_result = None
    roberta_result = None
    text = ""
    if request.method == 'POST':
        text = request.form['text']
        svm_result = analyze(text)
        roberta_result = combined_analysis(text)
    return render_template('index.html', text=text, svm_result=svm_result, roberta_result=roberta_result)

if __name__ == '__main__':
    app.run(debug=True)