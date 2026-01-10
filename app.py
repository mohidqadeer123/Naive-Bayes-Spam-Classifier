import streamlit as py
import pickle
import numpy as np 

# Load model and vectorizer
with open("model/model.pkl","rb") as f:
    model=pickle.load(f)

with open("model/vectorizer.pkl","rb") as f:
    vectorizer=pickle.load(f)


   
