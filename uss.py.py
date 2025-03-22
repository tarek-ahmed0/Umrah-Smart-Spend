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
            font-size: 1.1rem;  /* Smaller size for mobile */
            font-weight: bold;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .header-subtitle {
            font-family: 'Poppins', sans-serif;
            font-size: 0.9rem;  /* Smaller subtitle */
            color: #c34bff;
        }
        .divider {
            border-top: 1px solid #170225;
            margin: 15px 0;  /* Reduced margin */
        }
        .solid-border {
            border: 2px solid rgba(30, 10, 50);
            border-radius: 5px;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 15px;
            background-color: rgba(195, 75, 255, 0.5);
            width: 100%;
            box-sizing: border-box;
        }
        .column-label {
            font-family: 'Poppins', sans-serif;
            font-size: 0.85rem;  /* Smaller font for mobile */
            color: #6C63FF;
            font-weight: bold;
            margin-bottom: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("### :violet[🕋 Umrah Spending Predictor]")
st.markdown(":violet[Estimate your Umrah trip expenses based on your choices.]")
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

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

# Using markdown for labels with :violet[]
st.markdown(":violet[Month]")
month = col1.selectbox("", month_options)

st.markdown(":violet[Nationality]")
nationality = col2.selectbox("", ["Indian", "Turkish", "Egyptian", "Indonesian", "Jordanian", "Pakistani", "Sudanese"])

st.markdown(":violet[Age]")
age = col1.number_input("", min_value=1, max_value=70, step=1, value=30)

st.markdown(":violet[Umrah Type]")
umrah_type = col1.selectbox("", ["Individual", "Group"])

st.markdown(":violet[Accommodation Type]")
accommodation_type = col2.selectbox("", ["Hotel", "Apartment", "Relative's House"])

st.markdown(":violet[Transportation Mode]")
transportation_mode = col2.selectbox("", ["Bus", "Private Car", "Taxi", "On Foot"])

st.markdown(":violet[Stay Duration (days)]")
stay_duration = st.slider("", min_value=1, max_value=30, step=1, value=10)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Process features
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
            <h3 style="color: white; font-family: 'Poppins', sans-serif; font-size: 1.1rem;">
                🪙 Expected Budget : {prediction:,.0f} Riyal
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )
