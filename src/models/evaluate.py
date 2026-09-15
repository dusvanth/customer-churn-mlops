# src/models/evaluate.py

"""
Standalone evaluation script.
Loads a trained model and generates a classification report with metrics.
"""

import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)


def evaluate():
    """Load model and evaluate on test data."""
    # Load data
    df = pd.read_csv("data/processed/cleaned_telco.csv")
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Use the same split as training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    model = joblib.load("model.pkl")
    print(f"Loaded model: {type(model).__name__}")

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {
        "model": type(model).__name__,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = {
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
    }

    # Print results
    print("\n" + "=" * 50)
    print("📊 Evaluation Results")
    print("=" * 50)
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1 Score:  {metrics['f1_score']}")
    print(f"  ROC-AUC:   {metrics['roc_auc']}")
    print("\nConfusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    # Full classification report
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    # Save report
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "metrics.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Report saved to {report_path}")


if __name__ == "__main__":
    evaluate()
