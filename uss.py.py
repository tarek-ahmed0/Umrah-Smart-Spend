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
            font-size: 1rem; /* Reduced size for mobile */
            font-weight: bold;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .header-subtitle {
            font-family: 'Poppins', sans-serif;
            font-size: 0.875rem; /* Reduced size for mobile */
            color: #c34bff;
        }
        .divider {
            border-top: 1px solid #170225;
            margin: 15px 0;
        }
        .solid-border {
            border: 3px solid rgba(30, 10, 50);
            border-radius: 3px;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 15px;
            background-color: rgba(195, 75, 255, 0.5);
            width: 100%;
            box-sizing: border-box;
        }
        .violet-label label {
            font-family: 'Poppins', sans-serif;
            font-size: 1rem;
            font-weight: bold;
            color: #c34bff; /* Violet color for input labels */
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

with col1:
    month = st.selectbox("Month", month_options, key="month", help="Select your travel month", 
                         format_func=lambda x: f"🗓️ {x}", 
                         args=(st.session_state,), 
                         label_visibility="visible", 
                         placeholder="Select a month", 
                         help_visibility="visible", 
                         disabled=False, 
                         options=month_options, 
                         index=None, 
                         css_classes="violet-label")

    age = st.number_input("Age", min_value=1, max_value=70, step=1, value=30, key="age", 
                          css_classes="violet-label")

    umrah_type = st.selectbox("Umrah Type", ["Individual", "Group"], key="umrah_type", 
                              css_classes="violet-label")

with col2:
    nationality = st.selectbox("Nationality", ["Indian", "Turkish", "Egyptian", "Indonesian", "Jordanian", "Pakistani", "Sudanese"], 
                               key="nationality", css_classes="violet-label")

    accommodation_type = st.selectbox("Accommodation Type", ["Hotel", "Apartment", "Relative's House"], 
                                      key="accommodation_type", css_classes="violet-label")

    transportation_mode = st.selectbox("Transportation Mode", ["Bus", "Private Car", "Taxi", "On Foot"], 
                                       key="transportation_mode", css_classes="violet-label")

stay_duration = st.slider("Stay Duration (days)", min_value=1, max_value=30, step=1, value=10, 
                          key="stay_duration", css_classes="violet-label")

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
