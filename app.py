import joblib
import pandas as pd
import streamlit as st
import os

# Load the trained Random Forest model
model = joblib.load("optimized_rf.pkl")

# Feature order from training
feature_order = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Service_count'
]

st.set_page_config(page_title="Telco Customer Churn Predictor", layout="centered")

st.title("📊 Telco Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn based on their account and service information.")

# --- Step 1: Sidebar Inputs ---
st.sidebar.header("Customer Information")

tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=100, value=12, step=1)
monthlycharges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
total = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=1200.0, step=1.0)
service_count = st.sidebar.number_input("Number of Services Subscribed", min_value=0, max_value=7, value=3, step=1)

Contract = st.sidebar.selectbox("Contract Type", ['Monthly', "Yearly", "Two Year"])
payment = st.sidebar.selectbox("Payment Method", ['Electronic', "Mailed Check", "Bank Transfer", "Credit Card"])
Internet = st.sidebar.selectbox("Internet Service", ['DSL', "Fiber Optic", "No"])

# --- Step 2: Encode categorical inputs ---
contract_mapping = {"Monthly": 0, "Yearly": 1, "Two Year": 2}
contract_encoded = contract_mapping[Contract]

payment_mapping = {"Electronic": 0, "Mailed Check": 1, "Bank Transfer": 2, "Credit Card": 3}
payment_encoded = payment_mapping[payment]

internet_mapping = {"DSL": 0, "Fiber Optic": 1, "No": 2}
internet_encoded = internet_mapping[Internet]

# --- Step 3: Default values for other features ---
default_features = {
    'gender': 0,
    'SeniorCitizen': 0,
    'Partner': 0,
    'Dependents': 0,
    'PhoneService': 1,
    'MultipleLines': 0,
    'OnlineSecurity': 0,
    'OnlineBackup': 0,
    'DeviceProtection': 0,
    'TechSupport': 0,
    'StreamingTV': 0,
    'StreamingMovies': 0,
    'PaperlessBilling': 1
}

# --- Step 4: Merge user inputs with defaults ---
user_inputs = {
    'tenure': tenure,
    'MonthlyCharges': monthlycharges,
    'TotalCharges': total,
    'Service_count': service_count,
    'Contract': contract_encoded,
    'PaymentMethod': payment_encoded,
    'InternetService': internet_encoded
}

final_input = {**default_features, **user_inputs}

# --- Step 5: Create DataFrame and reorder columns ---
df_input = pd.DataFrame([final_input])
df_input = df_input[feature_order]

# --- Step 6: Predict button ---
st.markdown("---")
if st.button("Predict Churn"):
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    # Show result with nice UI
    if prediction == 1:
        st.error(f"⚠️ The customer is likely to churn! Probability: {probability:.2f}")
    else:
        st.success(f"✅ The customer is unlikely to churn. Probability: {probability:.2f}")

    # --- Optional: Record prediction ---
    record = df_input.copy()
    record['Churn_Prediction'] = prediction
    record['Churn_Probability'] = probability

    record_file = "prediction_records.csv"
    record.to_csv(record_file, mode='a', index=False, header=not os.path.exists(record_file))

    

# --- Step 7: Footer ---
st.markdown("---")
st.caption("Developed by InsightMonk | Powered by Random Forest Model")
