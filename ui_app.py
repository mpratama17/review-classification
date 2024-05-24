# Import convention
import streamlit as st
from util import predict
import pandas as pd

st.title("Review Text Classification")
text_input = st.text_input("Input Review")
if st.button("Predict"):
    prediction, preprocessed_text, tfidf_per_word = predict(text_input)
    st.write("Prediction: ", prediction)
    st.write("Preprocessed Text: ", preprocessed_text)
    # st.write("TFIDF: ", tfidf_review)
    st.write("TFIDF per word: ", tfidf_per_word)

# load dataset csv
df = pd.read_csv('dataset/Dataset User Review OTA decode.csv')
st.dataframe(df)
