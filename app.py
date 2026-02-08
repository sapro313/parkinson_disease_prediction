import streamlit as st
import pickle
import numpy as np

# Load assets
try:
    model = pickle.load(open('parkinson_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please run the training script first!")

# Page Styling
st.set_page_config(page_title="Parkinson's AI Diagnostic", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Parkinson's Disease Prediction")
st.info("This AI tool analyzes vocal acoustic parameters to assist in early detection.")

# Organise inputs into categories
st.subheader("Patient Vocal Measurements")
tab1, tab2, tab3 = st.tabs(["Frequency Metrics", "Jitter & Shimmer", "Complexity Measures"])

user_inputs = {}

with tab1:
    st.write("### Fundamental Frequencies")
    user_inputs['MDVP:Fo(Hz)'] = st.number_input("Average Frequency (Fo)", value=150.0)
    user_inputs['MDVP:Fhi(Hz)'] = st.number_input("Max Frequency (Fhi)", value=200.0)
    user_inputs['MDVP:Flo(Hz)'] = st.number_input("Min Frequency (Flo)", value=100.0)

with tab2:
    st.write("### Variation Measures")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Jitter (Frequency variation)**")
        for f in [f for f in features if 'Jitter' in f or 'RAP' in f or 'PPQ' in f or 'DDP' in f]:
            user_inputs[f] = st.number_input(f, value=0.005, format="%.5f")
    with col2:
        st.write("**Shimmer (Amplitude variation)**")
        for f in [f for f in features if 'Shimmer' in f or 'APQ' in f or 'DDA' in f]:
            user_inputs[f] = st.number_input(f, value=0.03, format="%.5f")

with tab3:
    st.write("### Tone & Signal Complexity")
    for f in features:
        if f not in user_inputs:
            user_inputs[f] = st.number_input(f, value=0.5, format="%.5f")

# Prediction Logic
st.divider()
if st.button("Analyze Results"):
    # Ensure inputs are in the correct order based on features.pkl
    ordered_input = [user_inputs[f] for f in features]
    input_data = np.array(ordered_input).reshape(1, -1)
    
    # Transform and Predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0]

    if prediction[0] == 1:
        st.error(f"### Prediction: Parkinson's Detected")
        st.progress(probability[1])
        st.write(f"Confidence: **{probability[1]:.2%}**")
        st.warning("Please consult a medical professional for a formal diagnosis.")
    else:
        st.success(f"### Prediction: Healthy / No Parkinson's")
        st.progress(probability[0])
        st.write(f"Confidence: **{probability[0]:.2%}**")

st.caption("Note: This tool is for educational purposes and should not replace professional medical advice.")