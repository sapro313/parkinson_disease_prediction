import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os


DOCTOR_USERNAME = "doctor"
DOCTOR_PASSWORD = "1234"

RESULT_FILE = "parkinson_results.csv"

st.set_page_config(page_title="Secure Parkinson App", layout="centered")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:

    st.title("🔐 Doctor Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == DOCTOR_USERNAME and password == DOCTOR_PASSWORD:
            st.session_state["authenticated"] = True
            st.success("Login successful! Loading application...")
            st.rerun()   
        else:
            st.error("Invalid credentials! Try again.")

    st.stop()   




try:
    model = pickle.load(open('parkinson_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please run the training script first!")


def save_results(user_inputs, prediction, probability):
    data = user_inputs.copy()
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["prediction"] = int(prediction[0])
    data["confidence"] = float(max(probability))

    df = pd.DataFrame([data])

    if os.path.exists(RESULT_FILE):
        df.to_csv(RESULT_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULT_FILE, index=False)



st.title("🧠 Parkinson's Disease Prediction (Secure Fog System)")

tab_predict, tab_results = st.tabs(["🔍 Predict", "📊 View Stored Results"])


with tab_predict:

    st.subheader("Patient Vocal Measurements")
    user_inputs = {}

    tab1, tab2, tab3 = st.tabs(["Frequency", "Jitter & Shimmer", "Complexity"])

    with tab1:
        user_inputs['MDVP:Fo(Hz)'] = st.number_input("Average Frequency (Fo)", value=150.0)
        user_inputs['MDVP:Fhi(Hz)'] = st.number_input("Max Frequency (Fhi)", value=200.0)
        user_inputs['MDVP:Flo(Hz)'] = st.number_input("Min Frequency (Flo)", value=100.0)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            for f in [f for f in features if 'Jitter' in f or 'RAP' in f or 'PPQ' in f or 'DDP' in f]:
                user_inputs[f] = st.number_input(f, value=0.005, format="%.5f")

        with col2:
            for f in [f for f in features if 'Shimmer' in f or 'APQ' in f or 'DDA' in f]:
                user_inputs[f] = st.number_input(f, value=0.03, format="%.5f")

    with tab3:
        for f in features:
            if f not in user_inputs:
                user_inputs[f] = st.number_input(f, value=0.5, format="%.5f")

    if st.button("Analyze Results"):
        ordered_input = [user_inputs[f] for f in features]
        input_data = np.array(ordered_input).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0]

        if prediction[0] == 1:
            st.error("Parkinson's Detected")
            st.progress(probability[1])
        else:
            st.success("Healthy")
            st.progress(probability[0])

        save_results(user_inputs, prediction, probability)
        st.caption("Results securely stored in cloud file.")


with tab_results:

    if not os.path.exists(RESULT_FILE):
        st.warning("No results stored yet.")
    else:
        df = pd.read_csv(RESULT_FILE)
        st.dataframe(df)

        total = len(df)
        risky = df["prediction"].sum()
        healthy = total - risky

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("High Risk", int(risky))
        c3.metric("Healthy", int(healthy))

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "parkinson_results.csv",
            "text/csv"
        )

st.caption("""
Note: Educational tool only.  
Training data = parkinsons.csv  
New patient records = parkinson_results.csv  
Access control implemented via doctor login.
""")
