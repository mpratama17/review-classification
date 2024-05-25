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
def predict(input_text):
    review = input_text
    processed_review = preprocess(review)
    tfidf_review = tfidf_transformer.transform(processed_review)
    tfidf_review_reshaped = np.array(tfidf_review).reshape(1, -1) # 
    prediction = model.predict(tfidf_review_reshaped)
    label = 'Satisfied' if prediction[0] == 1 else 'Unhappy'
    tfidf_per_word = getTFIDF(tfidf_transformer.word_list, tfidf_review, processed_review.split())

    return label, processed_review, tfidf_per_word

def getTFIDF(word_list, tfidf_score, sentence):
    tfidf_per_word = {}
    for i, word in enumerate(sentence):
        for j, val in enumerate(word_list):
            if val == word:
                tfidf_per_word[word] = tfidf_score[j]
    return tfidf_per_word


# def getTFIDF(word_list, tfidf_score, sentence):
#     tfidf_sentence = [0] * len(sentence)
#     for i, word in enumerate(sentence):
#         for j, val in enumerate(word_list):
#             if val == word:
#                 tfidf_sentence[i] = tfidf_score[j]
#     return tfidf_sentence


