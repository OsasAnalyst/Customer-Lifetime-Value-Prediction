# Customer Lifetime Value Prediction — Lumora Commerce

A machine learning project built for a DTC e-commerce brand to predict how much each customer will spend over the next 3 months and assign them to a retention tier.

The goal is simple: stop treating every customer the same. Some customers are worth fighting for. Others aren't. This system tells you which is which.

---

## The Problem

Lumora Commerce was allocating the same retention budget to every customer regardless of their actual value. No system existed to predict future spend, which meant loyalty rewards, email flows, and winback campaigns were all misdirected.

A 5% improvement in retention can grow profits by 25 to 95%. But you can't improve what you can't measure.

---

## What This Project Does

- Predicts each customer's 3-month forward revenue using historical transaction data
- Assigns every customer to a value tier: Platinum, Gold, Silver, or At-Risk
- Serves predictions through a REST API that can plug directly into Klaviyo or any CRM
- Deployed with Docker so it runs consistently anywhere

---

## Dataset

Online Retail II — UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/502/online+retail+ii

Real transactions from a UK-based online retailer. About 500k rows covering December 2009 to December 2010. One sheet was used after cleaning.

Download the file and place it in `data/raw/` before running the notebooks.

---

## Project Structure
```
ltv-retention-project/
├── api/
│   ├── main.py               # FastAPI app
│   ├── schemas.py            # Request and response models
│   └── predict.py            # Model loading and prediction logic
├── data/
│   ├── raw/                  # Original dataset (not tracked in git)
│   └── processed/            # Cleaned and engineered features
├── models/
│   ├── ltv_model.pkl         # Trained Linear Regression weights
│   ├── scaler.pkl            # StandardScaler
│   └── feature_cols.pkl      # Feature column names
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_feature_selection.ipynb
│   └── 04_modelling.ipynb
├── src/
│   └── models/
│       └── scratch/
│           └── linear_regression.py   # Built from scratch
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Features Engineered

All features are built from the observation window only. No data from the prediction window is used during training.

| Feature | Description |
|---|---|
| Recency | Days since last purchase |
| Frequency | Number of distinct orders |
| Monetary | Total spend in observation window |
| Avg order value | Monetary divided by frequency |
| Avg days between orders | Average gap between purchases |
| Return rate | Proportion of returned line items |
| Unique products | Number of distinct products purchased |
| Unique months active | How many months the customer placed at least one order |
| Last month revenue | Revenue in the final month of the observation window |

Log transformation was applied to all features to handle right skew.

---

## Model Results

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Linear Regression (Scratch) | 2.856 | 2.548 | 0.217 |
| Random Forest | 2.899 | 2.533 | 0.193 |
| Decision Tree (Scratch) | 2.937 | 2.539 | 0.172 |
| Decision Tree (sklearn) | 2.944 | 2.546 | 0.168 |
| XGBoost | 2.950 | 2.539 | 0.164 |

Linear Regression from scratch was selected as the final model.

---

## Why R2 is Low and Why That's Honest

40% of customers had zero revenue in the prediction window. They bought during the observation period but never came back. A regression model treats zero as just another value to predict, but zero here means something different — it means the customer churned.

This zero-inflation problem, combined with only 12 months of training data and a 3-month prediction window, puts a real ceiling on R2. Longer transaction history and a 6 to 12 month prediction window would improve results significantly.

Despite the R2, the model captures directional signal. Higher-value customers consistently score higher than lower-value ones. The tier assignments and ranked output are actionable.

---

## LTV Tiers

| Tier | Action |
|---|---|
| Platinum | VIP loyalty rewards and early access campaigns |
| Gold | Upsell and subscription offers |
| Silver | Nurture sequences and cross-sell |
| At-Risk | Low-cost winback or deprioritise |

---

## Running Locally
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

API docs at: http://127.0.0.1:8000/docs

---

## Running with Docker
```bash
docker build -t lumora-ltv-api .
docker run -p 8000:8000 lumora-ltv-api
```

API docs at: http://127.0.0.1:8000/docs

---

## API Usage
```
POST /predict

{
  "log_Recency": 3.4,
  "log_Frequency": 2.1,
  "log_Monetary": 5.8,
  "log_avg_days_between_orders": 2.3,
  "log_return_rate": 0.0,
  "log_unique_products": 3.1,
  "log_unique_months_active": 1.6,
  "log_last_month_revenue": 4.2
}

Response:
{
  "predicted_ltv_3m": 284.50,
  "ltv_tier": "Gold",
  "recommended_action": "Upsell and subscription offers"
}
```

---

## Tech Stack

Python, FastAPI, Docker, scikit-learn, XGBoost, pandas, numpy, matplotlib, seaborn

---

## Contact

Built by Osaretin Idiagbonmwen — Data Analyst specialising in DTC e-commerce.

LinkedIn: https://www.linkedin.com/in/osaretin-idiagbonmwen-33ab85339  
Email: oidiagbonmwen@gmail.com
