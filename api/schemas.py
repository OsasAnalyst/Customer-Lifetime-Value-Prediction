from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    log_Recency: float
    log_Frequency: float
    log_Monetary: float
    log_avg_order_value: float
    log_avg_days_between_orders: float
    log_tenure_days: float
    log_return_rate: float
    log_unique_products: float
    log_unique_months_active: float
    log_last_month_revenue: float

class PredictionResponse(BaseModel):
    predicted_ltv_3m: float
    ltv_tier: str
    recommended_action: str