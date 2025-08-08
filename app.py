import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Load model and columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Page setup
st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")
st.title("Telecom Customer Churn Prediction App")

# Sidebar Navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to:", ["Predict Churn", "Bulk Upload", "Insights"])

# Page 1: Predict Churn
if page == "Predict Churn":
    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

    input_data = {
        "gender": gender,
        "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
        "Partner": Partner,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    df_input = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in df_input.select_dtypes(include="object").columns:
        df_input[col] = df_input[col].astype("category").cat.codes

    # Ensure all required columns exist
    for col in model_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[model_columns]

    if st.button("Predict"):
        prediction = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0][1]

        st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
        st.info(f"Churn Probability: {proba:.2%}")

        # Save logs
        input_data.update({
            "Prediction": "Churn" if prediction == 1 else "No Churn",
            "Probability": round(proba, 2),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        if not os.path.exists("user_logs.csv"):
            pd.DataFrame(columns=input_data.keys()).to_csv("user_logs.csv", index=False)

        pd.DataFrame([input_data]).to_csv("user_logs.csv", mode="a", header=False, index=False)

# Page 2: Bulk Upload
elif page == "Bulk Upload":
    st.subheader("Upload CSV File for Batch Churn Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        for col in data.select_dtypes(include="object").columns:
            data[col] = data[col].astype("category").cat.codes

        for col in model_columns:
            if col not in data.columns:
                data[col] = 0
        data = data[model_columns]

        predictions = model.predict(data)
        data["Churn Prediction"] = ["Yes" if pred == 1 else "No" for pred in predictions]
        st.write(data)

        st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv")

# Page 3: Logs
elif page == "Insights":
    st.subheader("User Prediction Logs")

    if os.path.exists("user_logs.csv"):
        logs = pd.read_csv("user_logs.csv")
        st.write(logs.tail(20))
        churn_rate = logs['Prediction'].value_counts(normalize=True).get('Churn', 0)
        st.metric("Churn Rate (User Logs)", f"{churn_rate:.2%}")
    else:
        st.info("No user data yet.")
