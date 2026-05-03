import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

# *** UPDATE THIS TO YOUR HF USERNAME ***
repo_id = "your-username/tourism-prediction-model"
model_filename = "xgboost_tourism_model.joblib"

model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction")
st.write("Predict the likelihood of a customer purchasing the package based on specific factors.")

age = st.number_input("Age", 18, 100, 30)
duration = st.number_input("Pitch Duration (mins)", 1.0, 50.0, 10.0)

if st.button("Predict"):
    st.info("In a full deployment, these inputs will be mapped to the trained model features to output a 0 or 1.")
