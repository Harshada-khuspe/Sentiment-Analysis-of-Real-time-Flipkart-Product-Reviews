from flask import Flask, render_template, request, jsonify
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier


app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = DecisionTreeClassifier(random_state=0)
vectorizer = TfidfVectorizer(max_features=2500)

# Load the model and vectorizer
model=joblib.load("model.joblib")  # You need to save your trained model using joblib or pickle
vectorizer=joblib.load("vectorizer.joblib")  # Similarly, load your vectorizer

# Function to preprocess input
def preprocess_input(text):
    text = re.sub(r'[^\w\s]', '', text)
    preprocessed_text = ' '.join(token.lower() for token in nltk.word_tokenize(text) if token.lower() not in stopwords.words('english'))
    return preprocessed_text

# Function to predict sentiment
def predict_sentiment(review):
    preprocessed_review = preprocess_input(review)
    vectorized_review = vectorizer.transform([preprocessed_review]).toarray()
    prediction = model.predict(vectorized_review)
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    sentiment = predict_sentiment(review)
    if sentiment == 1:
        result = "Positive review!"
    else:
        result = "Negative review."
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
