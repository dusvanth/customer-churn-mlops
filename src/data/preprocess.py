# src/data/preprocess.py

import pandas as pd
import os


def preprocess():
    """
    Load raw Telco Customer Churn data, clean it,
    encode features, and save the processed dataset.
    """
    # Load raw data
    raw_path = os.path.join("data", "raw", "telco_customer_churn.csv")
    df = pd.read_csv(raw_path)

    # Drop customerID (not a feature)
    df = df.drop("customerID", axis=1)

    # Fix TotalCharges: some rows have blank strings instead of numbers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode binary categorical columns (Yes/No → 1/0)
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn"
    ]
    for col in binary_cols:
        if col == "gender":
            df[col] = df[col].map({"Female": 1, "Male": 0})
        else:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # Encode multi-level categoricals via one-hot encoding
    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaymentMethod"
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Convert all boolean columns to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Save processed data
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "cleaned_telco.csv")
    df.to_csv(processed_path, index=False)

    print(f"✅ Preprocessing complete. Saved to {processed_path}")
    print(f"   Shape: {df.shape}")


if __name__ == "__main__":
    preprocess()
