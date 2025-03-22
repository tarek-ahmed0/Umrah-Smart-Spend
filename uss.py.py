import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

        .header-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2.5rem;
            font-weight: bold;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header-subtitle {
            font-family: 'Poppins', sans-serif;
            font-size: 1.1rem;
            color: #c34bff;
        }
        .divider {
            border-top: 1px solid #170225;
            margin: 20px 0;
        }
        .solid-border {
            border: 3px solid rgba(30, 10, 50);
            border-radius: 3px;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            background-color: rgba(195, 75, 255, 0.5); /* Updated color with 50% opacity */
            width: 100%;
            height: 100%;
            box-sizing: border-box;
        }
        .column-label {
            font-family: 'Poppins', sans-serif;
            font-weight: bold;
            font-size: 1.1rem;
            color: #6C63FF;
            margin-bottom: 10px;
        }
        .column-container {
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="header-title">🕋 Umrah Spending Predictor</div>
    <div class="header-subtitle">Estimate your Umrah trip expenses based on your choices.</div>
    <div class="divider"></div>
    """,
    unsafe_allow_html=True
)

with open("umrah_spending_prediction.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("encoder.pkl", "rb") as enc_file:
    encoder = pickle.load(enc_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

col1, col2 = st.columns(2)
month_options = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December"
]
month = col1.selectbox("Month", month_options)
nationality = col2.selectbox("Nationality", ["Indian", "Turkish", "Egyptian", "Indonesian", "Jordanian", "Pakistani", "Sudanese"])
age = col1.number_input("Age", min_value = 1, max_value = 70, step = 1, value = 30)
umrah_type = col1.selectbox("Umrah Type", ["Individual", "Group"])
accommodation_type = col2.selectbox("Accommodation Type", ["Hotel", "Apartment", "Relative's House"])
transportation_mode = col2.selectbox("Transportation Mode", ["Bus", "Private Car", "Taxi", "On Foot"])
stay_duration = st.slider("Stay Duration (days)", min_value = 1, max_value = 30, step = 1, value = 10)

st.divider()

cat_features = pd.DataFrame([[month, nationality, umrah_type, accommodation_type, transportation_mode]],
                            columns=["Month", "Nationality", "Umrah_Type", "Accommodation_Type", "Transportation_Mode"])

encoded_cats = encoder.transform(cat_features)

if hasattr(encoded_cats, "toarray"):
    encoded_cats = encoded_cats.toarray()

full_features = np.hstack((encoded_cats, np.array([[age, stay_duration]])))
scaled_input = scaler.transform(full_features)

if st.button("Predict Spending"):
    prediction = model.predict(scaled_input)[0]
    prediction = abs(prediction)
    
    st.markdown(
        f"""
        <div class="solid-border">
            <h3 style="color: white; font-family: 'Poppins', sans-serif;">🪙 Expected Budget : {prediction:,.0f} Riyal</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
