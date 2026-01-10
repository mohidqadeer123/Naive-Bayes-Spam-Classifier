import streamlit as st 
import pickle
import numpy as np  

# Load model and vectorizer
with open("model/model.pkl","rb") as f:
    model=pickle.load(f)

with open("model/vectorizer.pkl","rb") as f:
    vectorizer=pickle.load(f)

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

   
