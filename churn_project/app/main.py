from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts the probability that a telecom customer will churn.",
    version="1.0.0"
)

# Load model pipeline (preprocessing + classifier bundled together)
model = joblib.load("churn_model.pkl")
col_info = joblib.load("columns.pkl")


class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100, description="Months as a customer")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check"
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5, "TotalCharges": 420.0
            }
        }


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerInput):
    input_df = pd.DataFrame([customer.dict()])
    # Ensure column order matches training
    input_df = input_df[col_info["all_cols"]] if all(
        c in input_df.columns for c in col_info["all_cols"]
    ) else input_df

    prob = model.predict_proba(input_df)[0][1]
    pred = int(prob >= 0.5)

    return {
        "churn_prediction": "Yes" if pred == 1 else "No",
        "churn_probability": round(float(prob), 4)
    }
