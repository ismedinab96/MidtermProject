import streamlit as st
import pickle
import numpy as np

st.title("🛍️ ML Zoomcamp Cohort 2025 ")
st.write("Upload your details to predict product rating!")
st.write("Elaborated by Iver Samuel Medina Balboa - ZoomCap ML Bootcamp - IA student from Computer Science at UMSA")
st.write("Tania Mujica My love for support ❤️")


with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/dv.pkl", "rb") as f:
    dv = pickle.load(f)

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)
    category = st.selectbox("Category", ["Clothing", "Electronics", "Books", "Home", "Beauty"])
    submit = st.form_submit_button("Prediction")

    if submit:
        input_data = {"gender": gender, "age": age, "category": category}
        X = dv.transform([input_data])
        pred = model.predict(X)[0]
        st.success(f"⭐ Review Rating: {pred:.2f}")
