import joblib
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'])

app = Flask(__name__)


model = joblib.load("modelrev.pkl")
vec = joblib.load("vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
en_stopwords = stopwords.words('english')

def prediction(sentence, model):
    labels = ["negative", "postive"]
    sentence = sentence.lower()
    punct = re.sub(r"([^\w\s])", "", sentence)
    no_stop_words = " ".join([word for word in punct.split() if word not in en_stopwords])
    tokenizer = word_tokenize(no_stop_words)
    bag = np.array(tokenizer)
    feature = vec.transform(bag)
    prediction =model.predict(feature)
    return  labels[prediction[0]]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            text = request.json['text']
        else:
            text = request.form['text']

        result = prediction(text, model)
        print(result)
        if request.is_json:
            return jsonify({
                'status': 'success',
                'prediction': result
            })
        else:
            return render_template('result.html',
                                   text=text,
                                   prediction=result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True)