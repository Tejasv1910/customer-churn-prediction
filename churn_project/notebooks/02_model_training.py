"""
Customer Churn Prediction - Model Training & Comparison
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import joblib

df = pd.read_csv('../data/churn_cleaned.csv')

X = df.drop('Churn', axis=1)
y = df['Churn']

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
])

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=200, eval_metric='logloss', random_state=42,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1])
    )
}

results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1': f1_score(y_test, preds),
        'ROC-AUC': roc_auc_score(y_test, probs)
    }
    results.append(metrics)
    fitted_pipelines[name] = pipe
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != 'Model':
            print(f"{k}: {v:.4f}")

results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
print("\n" + "=" * 60)
print("MODEL COMPARISON (sorted by F1)")
print("=" * 60)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_pipeline = fitted_pipelines[best_model_name]
print(f"\nBest model: {best_model_name}")

# Detailed report for best model
preds = best_pipeline.predict(X_test)
print("\nClassification Report (Best Model):")
print(classification_report(y_test, preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

# Save best pipeline (preprocessing + model bundled together)
joblib.dump(best_pipeline, '../app/churn_model.pkl')
print(f"\nSaved best pipeline ({best_model_name}) to app/churn_model.pkl")

# Save column info for the API
joblib.dump({'cat_cols': cat_cols, 'num_cols': num_cols, 'all_cols': list(X.columns)}, '../app/columns.pkl')
results_df.to_csv('../app/model_comparison.csv', index=False)
