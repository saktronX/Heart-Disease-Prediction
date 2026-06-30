import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's clinical details below.")

age = st.number_input("Age", 20, 100, 50)

sex = st.selectbox("Sex", ["Male", "Female"])

dataset = st.selectbox("Dataset", [0, 1, 2, 3])

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3],
    help="0: Typical Angina | 1: Atypical Angina | 2: Non-anginal Pain | 3: Asymptomatic"
)

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 250, 120)

chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

restecg = st.selectbox(
    "Resting ECG",
    [0, 1, 2]
)

thalach = st.number_input("Maximum Heart Rate Achieved", 50, 250, 150)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

if st.button("Predict"):

    sex = 1 if sex == "Male" else 0

    input_data = pd.DataFrame([[
        age,
        sex,
        dataset,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]], columns=[
        "age",
        "sex",
        "dataset",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalch",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
        confidence = probability[1]
    else:
        st.success("✅ No Heart Disease Detected")
        confidence = probability[0]

    st.write(f"**Confidence:** {confidence * 100:.2f}%")
    st.progress(float(confidence))