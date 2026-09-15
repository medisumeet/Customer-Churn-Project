# 📊 Telco Customer Churn Prediction

A Streamlit app that predicts whether a telecom customer is likely to
churn, based on their account and service details. Built end-to-end —
from statistical exploration to a deployed, interactive prediction
tool — on the
[IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

**Live app:** https://churnpredictor-ms.streamlit.app

## Highlights

- 📈 **Statistically grounded EDA** — every major driver of churn
  (contract type, payment method, tenure, service bundle) was validated
  with chi-square tests, not just visual inspection, before being called
  a "finding."
- 🧠 **Three models compared, one tuned properly** — Logistic
  Regression, Decision Tree, and Random Forest were benchmarked
  head-to-head, then the Random Forest was optimized with 5-fold
  `GridSearchCV` on F1 score.
- 🎯 **Solid predictive performance on an imbalanced target** — the
  tuned model reaches an F1-score of ~0.61 on the churn class, in line
  with published benchmarks for this dataset.
- 💡 **Actionable business insights, not just metrics** — the analysis
  translates model output into concrete findings, e.g. Electronic
  Check users churn at nearly 3x the rate of customers on automatic
  billing, and bundling Online Security / Backup / Tech Support cuts
  churn risk substantially.
- 🚀 **Shipped as a real, interactive tool** — rather than stopping at
  a notebook, the model is wrapped in a Streamlit app so anyone (not
  just someone reading code) can score a hypothetical customer in
  real time.

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

## Key findings

- **Payment method matters a lot.** Electronic Check users churn at
  ~45% — nearly 3x the rate of customers on automatic billing methods.
- **Contract length is a strong stabilizer.** Month-to-month customers
  churn far more than annual or two-year contract holders, confirming
  that commitment length tracks directly with loyalty.
- **Service bundling reduces risk.** Customers without Online Security,
  Online Backup, or Tech Support are meaningfully more likely to churn
  than those with multiple services active.
- **Tenure, MonthlyCharges, and TotalCharges are the top predictive
  features**, consistent with both the statistical tests and the
  Random Forest's feature importances.

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
