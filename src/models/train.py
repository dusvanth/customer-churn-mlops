# src/models/train.py

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load processed data
df = pd.read_csv("data/processed/cleaned_telco.csv")

# Split features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable MLflow experiment tracking
mlflow.set_experiment("telco-churn")
with mlflow.start_run(run_name="LogisticRegression"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics manually
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")

    # Save model as pickle for DVC tracking
    joblib.dump(model, "model.pkl")

    print("✅ Model training complete. Metrics logged to MLflow.")
    print(f"   Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
