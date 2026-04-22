import os
import pickle
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.title("Titanic Survival Predictor")
st.write("Enter passenger details:")

pclass = st.selectbox(
    "Passenger Class (Pclass)",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

age = st.number_input("Age", 0.0, 100.0, 25.0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)

if st.button("Predict Survival"):

    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "Fare": fare
    }])

    pred = model.predict(input_df)[0]

    label = "Will Survive ✅" if pred == 1 else "Will Not Survive ❌"

    st.subheader(label)
