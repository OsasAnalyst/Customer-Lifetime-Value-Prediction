import numpy as np
import joblib 
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.scratch.linear_regression import LinearRegressionScratch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load from weights
model = LinearRegressionScratch.load_from_weights(
    weights_path=os.path.join(BASE_DIR, '..', 'models', 'lr_weights.pkl'),
    bias_path=os.path.join(BASE_DIR, '..', 'models', 'lr_bias.pkl')
)

scaler = joblib.load(os.path.join(BASE_DIR, "..", "models", "scaler.pkl"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "..", "models", "feature_cols.pkl"))

P80 = 183.40000000000012
P50 = 34.79
P20 = 9.25


def assign_tier(ltv: float):
    if ltv >= P80:
        return "Platinum"
    elif ltv >= P50:
        return "Gold"
    elif ltv >= P20:
        return "Silver"
    else:
        return "At-Risk"
    

def get_recommendation(tier: str):
    action_map = {
        "Platinum": "VIP loyalty rewards + early access campaigns",
        "Gold": "Upsell and subscription offers",
        "Silver": "Nurture sequences + cross-sell",
        "At-Risk": "Low-cost winback or deprioritise"
    }
    return action_map[tier]


def predict_ltv(features: dict):
    X = np.array([[features[col] for col in feature_cols]])
    X_scaled = scaler.transform(X)
    log_pred = model.predict(X_scaled)[0]
    ltv = float(np.expm1(log_pred))
    ltv = max(0.0, round(ltv, 2))

    tier = assign_tier(ltv)
    action = get_recommendation(tier)

    return {
        'predicted_ltv_3m': ltv,
        'ltv_tier': tier,
        'recommended_action': action
    }