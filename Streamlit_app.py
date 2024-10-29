import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
model = pickle.load(open('finalized_model_SVM.sav', 'rb'))
scaler = pickle.load(open('scaler_SVM.sav', 'rb'))

# Add background image using CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-photo/beautiful-beach-galapagos-islands-ecuador_53876-101404.jpg?t=st=1730198043~exp=1730201643~hmac=6109889851e55a2705ea45d05a7c024e671d55a3c92153a091f9a787f586ac8b&w=1380");
        background-size: cover;
        background-position: center;
        height: 100vh;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Center the title using Markdown
st.markdown("<h1 style='text-align: center;'>Phi Estimation of Silty Sand</h1>", unsafe_allow_html=True)

# Center-aligned, larger description text
st.markdown(
    "<p style='text-align: center; font-size:20px;'>Estimate the angle of internal friction (Phi) for Silty Sand samples based on SPT-N60 and grain size analysis.</p>",
    unsafe_allow_html=True
)

# Create two columns for the input fields
col1, col2 = st.columns(2)

with col1:
    SPTN = st.number_input("SPT-N with Field Correction (SPT-N60)", min_value=1.0, max_value=30.0, step=1.0)
    d30 = st.number_input("D30", min_value=0.0, max_value=1.0, step=0.001)
    silt = st.number_input("Percentage of Silt", min_value=0.0, max_value=80.0, step=0.01)

with col2:
    depth = st.number_input("Depth of Sample Collection (m)", min_value=0.0, max_value=80.0, step=0.5)
    d60 = st.number_input("D60", min_value=0.0, max_value=1.0, step=0.01)

# Prediction
if st.button("Predict"):
    # Prepare and scale input data
    sample_data = np.array([SPTN, depth, d30, d60, silt]).reshape(1, -1)
    scaled_data = scaler.transform(sample_data)
    prediction = model.predict(scaled_data)
    
    # Display result
    st.success(f"Predicted Angle of Internal Friction (Phi): {prediction[0]:.2f}°")

