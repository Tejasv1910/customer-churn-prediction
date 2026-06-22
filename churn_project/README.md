# Customer Churn Prediction System

An end-to-end machine learning system that predicts whether a telecom customer
is likely to churn, served via a FastAPI REST API and containerized with Docker.

## Problem Statement

Customer churn (customers leaving for a competitor) is one of the most costly
problems for subscription-based businesses. This project builds a classifier
that flags high-risk customers from account and service usage data, so a
business could proactively target them with retention offers.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(IBM sample dataset, via Kaggle) — 7,043 customers, 20 features (demographics,
account info, services subscribed), with a binary churn label.
Class distribution: ~73.5% No-churn / 26.5% Churn (imbalanced).

## Approach

1. **EDA & Cleaning** — Identified 11 records with blank `TotalCharges`,
   traced them to customers with `tenure = 0` (not yet billed), and imputed
   accordingly. Found `tenure` to be the strongest individual predictor of
   churn (correlation: -0.35).
2. **Preprocessing** — One-hot encoded 15 categorical features, standardized
   4 numerical features, using an sklearn `ColumnTransformer` + `Pipeline` so
   preprocessing and model are bundled into a single deployable artifact.
3. **Modeling** — Trained and compared 3 classifiers (Logistic Regression,
   Random Forest, XGBoost) with class-balancing, evaluated on Accuracy,
   Precision, Recall, F1, and ROC-AUC.
4. **Model Selection** — Logistic Regression was selected as the best model
   (F1: 0.614, ROC-AUC: 0.842), outperforming the more complex tree-based
   models — a reminder that simpler, well-regularized models can win on
   tabular data with class imbalance.
5. **Serving** — Exposed the trained pipeline through a FastAPI `/predict`
   endpoint with full request validation (Pydantic).
6. **Deployment** — Containerized with Docker for environment-independent,
   reproducible deployment.

## Results

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|----------------------|----------|-----------|--------|-------|---------|
| **Logistic Regression** | 0.739    | 0.505     | 0.783  | 0.614 | **0.842** |
| XGBoost              | 0.759    | 0.541     | 0.620  | 0.578 | 0.807   |
| Random Forest        | 0.789    | 0.627     | 0.503  | 0.558 | 0.824   |

> Recall was prioritized for the churn class: a missed churner (false negative)
> is costlier to the business than a false alarm, since the cost of an
> unnecessary retention offer is far lower than losing a customer.

## Tech Stack

Python · Pandas · Scikit-learn · XGBoost · FastAPI · Pydantic · Docker

## Project Structure

```
churn_project/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   ├── 01_eda_preprocessing.py
│   └── 02_model_training.py
├── app/
│   ├── main.py              # FastAPI app
│   ├── churn_model.pkl      # trained sklearn pipeline
│   ├── columns.pkl          # column metadata for inference
│   ├── requirements.txt
│   └── Dockerfile
└── README.md
```

## Running Locally

### Option 1 — Without Docker
```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload
```
Visit `http://localhost:8000/docs` for interactive Swagger UI.

### Option 2 — With Docker
```bash
cd app
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5, "TotalCharges": 420.0
  }'
```

**Response:**
```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.863
}
```

## Live Demo

🔗 _[Add your deployed URL here once deployed to Render / HF Spaces]_

## Author

Tejasv Rathore
