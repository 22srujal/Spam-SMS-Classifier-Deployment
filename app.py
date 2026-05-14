from __future__ import annotations

from pathlib import Path
import pickle

from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'spam-sms-mnb-model.pkl'
VECTORIZER_PATH = BASE_DIR / 'cv-transform.pkl'

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
with MODEL_PATH.open('rb') as model_file:
	classifier = pickle.load(model_file)

with VECTORIZER_PATH.open('rb') as vectorizer_file:
	cv = pickle.load(vectorizer_file)


app = Flask(__name__)

@app.route('/')
def home():

    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')
    vect = cv.transform([message])
    prediction = int(classifier.predict(vect)[0])
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':

    app.run(debug=True)