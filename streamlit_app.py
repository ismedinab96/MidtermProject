import streamlit as st
import pickle
import numpy as np

st.title("🛍️ Midterm ML Zoomcamp Predictor")
st.write("Sube los datos del cliente para predecir la calificación de reseña.")


with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/dv.pkl", "rb") as f:
    dv = pickle.load(f)

with st.form("prediction_form"):
    gender = st.selectbox("Género", ["Male", "Female"])
    age = st.slider("Edad", 18, 100, 30)
    category = st.selectbox("Categoría", ["Clothing", "Electronics", "Books", "Home", "Beauty"])
    submit = st.form_submit_button("Predecir")

    if submit:
        input_data = {"gender": gender, "age": age, "category": category}
        X = dv.transform([input_data])
        pred = model.predict(X)[0]
        st.success(f"⭐ Predicción de calificación: {pred:.2f}")
