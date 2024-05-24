from flask import Flask, request, render_template
import pickle
import re
import numpy as np
from multinomial import MultinomialNB
from preprocessing import preprocess
import pandas as pd



# Load the trained model and TFIDF vectorizer
with open('pickle/model_91_new.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('pickle/tfidf_transformer.pkl', 'rb') as transformer_file:
    tfidf_transformer = pickle.load(transformer_file)

# Load dataset
df = pd.read_csv('dataset/Dataset User Review OTA decode.csv')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        processed_review = preprocess(review)
        tfidf_review = tfidf_transformer.transform(processed_review)
        tfidf_review_reshaped = np.array(tfidf_review).reshape(1, -1) # 
        prediction = model.predict(tfidf_review_reshaped)
        label = 'Satisfied' if prediction[0] == 1 else 'unhappy'
        return render_template('index.html', prediction=label)

@app.route('/dataset')
def show_dataset():
    search_query = request.args.get('search', '')
    if search_query:
        filtered_df = df[df['content'].str.contains(search_query, case=False, na=False)]
    else:
        filtered_df = df
    dataset = filtered_df.to_dict(orient='records')
    return render_template('dataset.html', dataset=dataset)

if __name__ == "__main__":
    app.run(debug=True)