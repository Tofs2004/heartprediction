import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Title
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter the patient's health details below to predict the likelihood of heart disease.")

# Load the trained model
MODEL_PATH = 'xgb_heartdisease_model.pkl'

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"❌ Model file '{MODEL_PATH}' not found. Please upload or train a model first.")
    st.stop()

# Input form
with st.form("heart_disease_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resting ECG results (restecg)", options=[0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
    oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    slope = st.selectbox("Slope of the ST segment (slope)", options=[0, 1, 2])
    ca = st.selectbox("Major vessels colored by fluoroscopy (ca)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

    submitted = st.form_submit_button("Predict")

# Predict
if submitted:
    features = np.array([
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]).reshape(1, -1)

    try:
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.error("💔 The patient is **likely to have heart disease**.")
        else:
            st.success("❤️ The patient is **not likely to have heart disease**.")
    except Exception as e:
        st.exception(f"Prediction failed: {e}")
