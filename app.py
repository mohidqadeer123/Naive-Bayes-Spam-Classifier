import streamlit as st 
import pickle
import numpy as np  
from joblib import load

# Load model and vectorizer
model = load("model/model.joblib")
vectorizer = load("model/vectorizer.joblib")

st.set_page_config(page_title="Spam Classifier",layout="centered")

st.title("📩 Spam Detection App")
st.write("Classifies messages as **Spam** or **Ham** using Multinomial Naive Bayes")

# User input
user_input=st.text_area("Enter the Message")

if st.button("Predict"):
    if user_input.strip() == "": 
        st.warning("Please enter a message:")

    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed).max()

        if prediction == 1:
            st.error(f"🚨 SPAM ({probability:.2%} confidence)")

        else:
            st.success(f"✅ HAM ({probability:.2%} confidence)")

   
