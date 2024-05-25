import pandas as pd
import streamlit as st

# load dataset from csv
df = pd.read_csv('dataset/Dataset User Review OTA decode.csv')
st.write(df.head())