# Telco Customer Churn Prediction

A Streamlit app that predicts whether a telecom customer is likely to
churn, based on their account and service details. Built on the
[IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

**Live app:** _add your Streamlit Cloud URL here once deployed_

## What's in this repo

| File | Purpose |
|---|---|
| `churn-analysis.ipynb` | Exploratory data analysis and model training |
| `app.py` | Streamlit app that takes 7 key inputs and returns a churn prediction |
| `optimized_rf.pkl` | The trained Random Forest model |
| `requirements.txt` | Python dependencies needed to run the app |

## Approach

1. **EDA** — used chi-square tests and grouped comparisons to check which
   account/service attributes were associated with churn (e.g. contract
   type, payment method, tenure, service usage).
2. **Preprocessing** — cleaned the dataset, manually encoded binary
   yes/no fields, and label-encoded multi-category fields (`Contract`,
   `PaymentMethod`, `InternetService`).
3. **Modeling** — compared Logistic Regression, Decision Tree, and Random
   Forest; tuned the Random Forest with `GridSearchCV` (5-fold CV,
   optimized for F1 score).
4. **App** — the UI takes the 7 features used as input (tenure, monthly
   charges, total charges, number of add-on services, contract type,
   payment method, internet service) and uses fixed defaults for the
   remaining features the model was trained on.

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser (usually at `localhost:8501`).

## Deploying on Streamlit Community Cloud

1. Make sure `optimized_rf.pkl` and `requirements.txt` are committed to
   this repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, and point it at this repo's `main` branch
   and `app.py`.
3. Deploy.

## Limitations

- The UI exposes only 7 of the features the model was trained on; the
  rest are held at fixed defaults.
- Trained on a static, historical, public dataset.