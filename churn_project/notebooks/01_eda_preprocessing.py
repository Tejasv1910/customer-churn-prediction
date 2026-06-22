"""
Customer Churn Prediction - EDA & Preprocessing
"""
import pandas as pd
import numpy as np

df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("=" * 50)
print("STEP 1: Basic Info")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True)}\n")

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"Missing TotalCharges (after coercion): {df['TotalCharges'].isnull().sum()}")
# These are tenure=0 customers -> haven't been billed yet -> fill with 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Drop ID column - not predictive
df = df.drop('customerID', axis=1)

print("\n" + "=" * 50)
print("STEP 2: Encode target")
print("=" * 50)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("\n" + "=" * 50)
print("STEP 3: Identify column types")
print("=" * 50)
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols.remove('Churn')
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical columns ({len(num_cols)}): {num_cols}")

print("\n" + "=" * 50)
print("STEP 4: Correlation of numeric features with churn")
print("=" * 50)
print(df[num_cols + ['Churn']].corr()['Churn'].sort_values(ascending=False))

# Save cleaned dataset
df.to_csv('../data/churn_cleaned.csv', index=False)
print("\nSaved cleaned dataset to data/churn_cleaned.csv")
print(f"Final shape: {df.shape}")
