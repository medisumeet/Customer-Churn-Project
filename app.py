import joblib
import pandas as pd
import streamlit as st

# Load the trained Random Forest model
model = joblib.load("optimized_rf.pkl")

# Save the exact training feature order (replace with your X_train.columns list)
feature_order = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Service_count'
]

st.title("Telco Customer Churn Prediction")

# --- Step 1: User Inputs ---
tenure = st.number_input("Enter Tenure (Months)", min_value=0, max_value=100, value=0, step=1)
monthlycharges = st.number_input("Enter Monthly Charges", min_value=0)
total = st.number_input("Enter Total Charges", min_value=0)
service_count = st.number_input("Enter the Service Count", min_value=0, max_value=3, step=1)

Contract = st.selectbox("Contract Type", ['Monthly', "Yearly", "Two Year"])
payment = st.selectbox("Payment Method", ['Electronic', "Mailed Check", "Bank Transfer", "Credit Card"])
Internet = st.selectbox("Internet", ['DSL', "Fiber Optic", "No"])

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

# --- Step 5: Create DataFrame ---
df_input = pd.DataFrame([final_input])

# --- Step 6: Reorder columns to match training ---
df_input = df_input[feature_order]

# --- Step 7: Prediction on button click ---
if st.button("Predict Churn"):
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    if prediction == 1:
        st.warning(f"The customer is likely to churn. Probability: {probability:.2f}")
        st.write("Customer is less likely to CHURN")
    else:
        st.success(f"The customer is unlikely to churn. Probability: {probability:.2f}")
