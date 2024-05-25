# Import convention
import streamlit as st
from util import predict
import pandas as pd
import time
from datetime import datetime
import numpy as np

st.title(":earth_asia: Review Classification.")
multi = """
    This is a simple text classification web app to predict a user review online travel agent,  
    by :blue-background[Mohammad Yoga Pratama]. &mdash; :bullettrain_side::bus::airplane:
    """
st.write(multi)

text_input = st.text_input("Input Review 👇", placeholder='Input review text here...', )
if st.button("Predict"):
    prediction, preprocessed_text, tfidf_per_word = predict(text_input)
    st.write("Prediction: ", prediction)
    st.write("Preprocessed Text: ", preprocessed_text)
    # st.write("TFIDF: ", tfidf_review)
    st.write("TFIDF per word: ", tfidf_per_word)

st.sidebar.title("Text Classification")




