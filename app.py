import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime

# ===========================
# SESSION STATE (MUST BE FIRST)
# ===========================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# ===========================
# SECURITY CREDENTIALS
# ===========================
DOCTOR_USERNAME = "doctor"
DOCTOR_PASSWORD = "1234"

# ===========================
# LOGIN SCREEN (BLOCKS APP)
# ===========================
if not st.session_state["authenticated"]:

    st.set_page_config(page_title="Doctor Login", layout="centered")

    st.title("🔐 Doctor Login Required")

    username = st.text_input("Username", key="uname")
    password = st.text_input("Password", type="password", key="pwd")

    if st.button("Login"):

        # Clean inputs (removes spaces + quotes)
        u = username.strip().replace('"','').replace("'", "")
        p = password.strip().replace('"','').replace("'", "")

        if u == DOCTOR_USERNAME and p == DOCTOR_PASSWORD:
            st.session_state["authenticated"] = True
            st.success("Login successful! Loading application...")
            st.rerun()
        else:
            st.error("Invalid credentials! Try again.")

    st.stop()   # IMPORTANT: stops app until login succeeds


# ===========================
# MAIN APPLICATION STARTS HERE
# ===========================

st.set_page_config(page_title="Parkinson's AI Diagnostic", layout="centered")

# Load assets
try:
    model = pickle.load(open('parkinson_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please run the training script first!")
    st.stop()

RESULT_FILE = "parkinson_results.csv"

# Create results file if not exists
if not os.path.exists(RESULT_FILE):
    pd.DataFrame(columns=["Timestamp", "Prediction", "Confidence"]).to_csv(RESULT_FILE, index=False)

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Parkinson's Disease Prediction")
st.info("This AI tool analyzes vocal acoustic parameters to assist in early detection.")

# Tabs
tab1, tab2 = st.tabs(["🔍 Predict", "📁 View Stored Results"])

# ===========================
# TAB 1 - PREDICTION PAGE
# ===========================
with tab1:

    st.subheader("Patient Vocal Measurements")

    user_inputs = {}

    st.write("### Fundamental Frequencies")
    user_inputs['MDVP:Fo(Hz)'] = st.number_input("Average Frequency (Fo)", value=150.0)
    user_inputs['MDVP:Fhi(Hz)'] = st.number_input("Max Frequency (Fhi)", value=200.0)
    user_inputs['MDVP:Flo(Hz)'] = st.number_input("Min Frequency (Flo)", value=100.0)

    st.write("### Other Acoustic Features")
    for f in features:
        if f not in user_inputs:
            user_inputs[f] = st.number_input(f, value=0.5, format="%.5f")

    st.divider()

    if st.button("Analyze Results"):

        ordered_input = [user_inputs[f] for f in features]
        input_data = np.array(ordered_input).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if prediction[0] == 1:
            result = "High Risk - Parkinson Detected"
            confidence = f"{probability[1]:.2%}"

            st.error("### 🚨 Parkinson's Detected")
            st.progress(probability[1])
            st.write(f"Confidence: **{confidence}**")
            st.warning("Please consult a medical professional.")

        else:
            result = "Healthy / No Parkinson"
            confidence = f"{probability[0]:.2%}"

            st.success("### ✅ Healthy / No Parkinson")
            st.progress(probability[0])
            st.write(f"Confidence: **{confidence}**")

        # Save result to CSV (acts as CLOUD STORAGE)
        df = pd.read_csv(RESULT_FILE)
        new_row = pd.DataFrame({
            "Timestamp": [timestamp],
            "Prediction": [result],
            "Confidence": [confidence]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(RESULT_FILE, index=False)

        st.success("Result saved to cloud storage (CSV).")


# ===========================
# TAB 2 - VIEW RESULTS
# ===========================
with tab2:

    st.subheader("📊 Stored Parkinson Results (Cloud Layer)")

    df = pd.read_csv(RESULT_FILE)
    st.dataframe(df)

    st.download_button(
        label="📥 Download Results CSV",
        data=df.to_csv(index=False),
        file_name="parkinson_results.csv",
        mime="text/csv"
    )

    st.write(f"Total records stored: **{len(df)}**")


st.caption("Note: This tool is for educational purposes and should not replace professional medical advice.")
