from fastapi import FastAPI, HTTPException
from api.schemas import CustomerFeatures, PredictionResponse
from api.predict import predict_ltv


app = FastAPI(
    title="Lumora Commerce — LTV Prediction API",
    description="Predicts 3-month customer lifetime value and assigns retention tier",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {
        "status":  "ok",
        "model":   "Linear Regression (Scratch)",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        features = customer.model_dump()
        result = predict_ltv(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))