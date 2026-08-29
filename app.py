import joblib
import pandas as pd
import streamlit as st

# --- Load model ---
model = joblib.load("optimized_rf.pkl")

feature_order = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Service_count'
]

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .main {
        padding-top: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        color: white;
        text-align: center;
    }
    
    .header-container h1 {
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .header-container p {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    .input-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border-left: 5px solid #667eea;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
    }
    
    .result-success {
        border-left: 5px solid #10b981;
    }
    
    .result-warning {
        border-left: 5px solid #f59e0b;
    }
    
    .result-danger {
        border-left: 5px solid #ef4444;
    }
    
    .feature-section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    .risk-indicator {
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    .risk-medium {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .risk-low {
        background-color: #dcfce7;
        color: #166534;
    }
    
    .button-container {
        text-align: center;
        padding: 2rem 0;
    }
    
    div[data-testid="stMetricValue"] { 
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        color: #6b7280;
        font-weight: 600;
    }
    
    .footer-text {
        text-align: center;
        font-size: 0.95rem;
        color: #6b7280;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 2px solid #e5e7eb;
    }
    
    .divider-custom {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-container">
        <h1>📊 Telco Customer Churn Predictor</h1>
        <p>🔮 AI-Powered Churn Risk Analysis · Predict Customer Retention</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="input-card">
        <div class="feature-section-title">📋 Enter Customer Details</div>
        <p style="color: #6b7280; margin-bottom: 1rem;">Analyze key account metrics to predict churn likelihood</p>
    </div>
""", unsafe_allow_html=True)

# --- Top predictive features as inputs ---
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("<div class='feature-section-title'>💰 Charges</div>", unsafe_allow_html=True)
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12, step=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1200.0, step=1.0)

with col2:
    st.markdown("<div class='feature-section-title'>📦 Services</div>", unsafe_allow_html=True)
    service_count = st.slider("Add-on Services", min_value=0, max_value=7, value=3)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col3:
    st.markdown("<div class='feature-section-title'>💳 Payment</div>", unsafe_allow_html=True)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

# --- Encoding maps ---
# Verified against the standard IBM Telco Customer Churn dataset's category
# order (sklearn LabelEncoder assigns codes alphabetically). Re-check against
# your own df_raw printout if your training data differs from the standard set.
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

# --- Defaults for everything not exposed in the UI ---
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
    'PaperlessBilling': 1,
}

user_inputs = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Service_count': service_count,
    'Contract': contract_map[contract],
    'PaymentMethod': payment_map[payment_method],
    'InternetService': internet_map[internet_service],
}

final_input = {**default_features, **user_inputs}
df_input = pd.DataFrame([final_input])[feature_order]

st.markdown("<div class='divider-custom'></div>", unsafe_allow_html=True)

_, mid, _ = st.columns([1, 2, 1])
with mid:
    predict_clicked = st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary", help="Click to analyze churn probability")

st.markdown("<div class='divider-custom'></div>", unsafe_allow_html=True)

if predict_clicked:
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    # Results display
    st.markdown("<h2 style='color: #667eea; text-align: center;'>✨ Prediction Results</h2>", unsafe_allow_html=True)
    
    result_col1, result_col2, result_col3 = st.columns([1.2, 1.5, 1.2], gap="large")

    with result_col1:
        if prediction == 1:
            st.markdown("""
                <div class="result-card result-danger" style="text-align: center; padding: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚠️</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #ef4444;">Likely to Churn</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-card result-success" style="text-align: center; padding: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">✅</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #10b981;">Unlikely to Churn</div>
                </div>
            """, unsafe_allow_html=True)

    with result_col2:
        st.markdown("""
            <div class="result-card" style="padding: 2rem;">
                <div style="font-size: 1rem; color: #6b7280; font-weight: 600; margin-bottom: 0.5rem;">Churn Probability Score</div>
        """, unsafe_allow_html=True)
        st.metric("Risk Probability", f"{probability:.0%}")
        st.progress(min(float(probability), 1.0), text=f"{probability:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col3:
        st.markdown("""
            <div class="result-card" style="padding: 2rem; text-align: center;">
                <div style="font-size: 1rem; color: #6b7280; font-weight: 600; margin-bottom: 1rem;">Risk Assessment</div>
        """, unsafe_allow_html=True)
        if probability > 0.7:
            st.markdown('<div class="risk-indicator risk-high">🔴 HIGH RISK</div>', unsafe_allow_html=True)
            st.caption("⚡ Urgent: Consider immediate retention outreach")
        elif probability > 0.4:
            st.markdown('<div class="risk-indicator risk-medium">🟡 MODERATE RISK</div>', unsafe_allow_html=True)
            st.caption("📍 Monitor closely and prepare retention strategies")
        else:
            st.markdown('<div class="risk-indicator risk-low">🟢 LOW RISK</div>', unsafe_allow_html=True)
            st.caption("✨ Customer appears stable and satisfied")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider-custom'></div>", unsafe_allow_html=True)
    
    with st.expander("🔍 See Detailed Input Data (including system defaults)"):
        st.markdown("""
            <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <p style="color: #6b7280; font-size: 0.9rem;">
                    ℹ️ This table shows all features sent to the model, including auto-populated default values for fields not exposed in the UI.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_input, use_container_width=True)

st.markdown("""
    <div class='footer-text'>
        <p><strong>🤖 Powered by Random Forest ML Model</strong></p>
        <p style="font-size: 0.85rem; color: #9ca3af; margin-top: 0.5rem;">
            Trained on IBM Telco Customer Churn Dataset · Predicts customer retention likelihood
        </p>
    </div>
""", unsafe_allow_html=True)