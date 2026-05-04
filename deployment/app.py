import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# *** UPDATE THIS TO YOUR HF USERNAME ***
repo_id        = "sandy1916/Tourism-Prediction-Space"
model_filename = "tourism_best_model.joblib"

# Load model from HF Model Hub
@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=repo_id, filename=model_filename)
    return joblib.load(path)

model = load_model()

# App UI
st.title("Wellness Tourism Package Prediction")
st.write("Predict whether a customer is likely to purchase the Wellness Tourism Package.")

st.header("Customer Details")
col1, col2 = st.columns(2)

with col1:
    age                    = st.number_input("Age", 18, 100, 35)
    city_tier              = st.selectbox("City Tier", [1, 2, 3])
    number_of_person       = st.number_input("Number of Persons Visiting", 1, 10, 2)
    preferred_star         = st.selectbox("Preferred Property Star", [3, 4, 5])
    number_of_trips        = st.number_input("Number of Trips per Year", 0, 20, 2)
    passport               = st.selectbox("Has Passport?", [0, 1])
    own_car                = st.selectbox("Owns a Car?", [0, 1])
    children_visiting      = st.number_input("Number of Children Visiting (<5 yrs)", 0, 5, 0)
    monthly_income         = st.number_input("Monthly Income (INR)", 5000, 100000, 25000)

with col2:
    pitch_satisfaction     = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    number_of_followups    = st.number_input("Number of Follow-ups", 0, 10, 3)
    duration_of_pitch      = st.number_input("Duration of Pitch (mins)", 1.0, 60.0, 15.0)
    type_of_contact        = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    occupation             = st.selectbox("Occupation", ["Salaried", "SmallBusiness", "LargeBusinesss", "FreeLancer"])
    gender                 = st.selectbox("Gender", ["Male", "Female"])
    marital_status         = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    designation            = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    product_pitched        = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

if st.button("Predict", type="primary"):
    # Build input record matching training feature schema
    input_dict = {
        "Age": age,
        "CityTier": city_tier,
        "NumberOfPersonVisiting": number_of_person,
        "PreferredPropertyStar": preferred_star,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children_visiting,
        "MonthlyIncome": monthly_income,
        "PitchSatisfactionScore": pitch_satisfaction,
        "NumberOfFollowups": number_of_followups,
        "DurationOfPitch": duration_of_pitch,
        # One-hot encoded fields (drop_first=True encoding from training)
        "TypeofContact_Self Enquiry": 1 if type_of_contact == "Self Inquiry" else 0,
        "Occupation_Small Business": 1 if occupation == "SmallBusiness" else 0,
        "Occupation_Large Business": 1 if occupation == "LargeBusinesss" else 0,
        "Occupation_FreeLancer": 1 if occupation == "FreeLancer" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "MaritalStatus_Married": 1 if marital_status == "Married" else 0,
        "MaritalStatus_Single": 1 if marital_status == "Single" else 0,
        "Designation_Manager": 1 if designation == "Manager" else 0,
        "Designation_Senior Manager": 1 if designation == "Senior Manager" else 0,
        "Designation_AVP": 1 if designation == "AVP" else 0,
        "Designation_VP": 1 if designation == "VP" else 0,
        "ProductPitched_Deluxe": 1 if product_pitched == "Deluxe" else 0,
        "ProductPitched_King": 1 if product_pitched == "King" else 0,
        "ProductPitched_Standard": 1 if product_pitched == "Standard" else 0,
        "ProductPitched_Super Deluxe": 1 if product_pitched == "Super Deluxe" else 0,
    }

    # Align with model's expected feature order
    input_df = pd.DataFrame([input_dict])
    expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else input_df.columns
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    st.divider()
    if prediction == 1:
        st.success(f"Prediction: This customer is LIKELY to purchase the Wellness Tourism Package.")
    else:
        st.warning(f"Prediction: This customer is UNLIKELY to purchase the Wellness Tourism Package.")

    if probability is not None:
        st.info(f"Purchase probability: {probability:.2%}")
