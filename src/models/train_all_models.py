# src/models/train_all_models.py

"""
Train and compare multiple ML models for churn prediction.
Models: Logistic Regression, Random Forest, XGBoost, LightGBM
Each model is logged as a separate MLflow run.
"""

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import os
import json


def load_data():
    """Load preprocessed data and split into train/test sets."""
    df = pd.read_csv("data/processed/cleaned_telco.csv")
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


def train_and_log(model, model_name, X_train, y_train, X_test, y_test, params=None):
    """Train a model and log everything to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_param("model_name", model_name)
        if params:
            mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, round(value, 4))

        # Log model
        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, "model")
        elif "LGBM" in model_name:
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        print(f"  {model_name}: accuracy={metrics['accuracy']:.4f}, "
              f"f1={metrics['f1_score']:.4f}, roc_auc={metrics['roc_auc']:.4f}")

    return model, metrics


def main():
    print("=" * 60)
    print("🚀 Training Multiple Models for Churn Prediction")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()
    print(f"\nDataset: {X_train.shape[0]} train / {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}\n")

    # Define models and their hyperparameters
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"max_iter": 1000, "solver": "lbfgs"},
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            "params": {"n_estimators": 200, "max_depth": 10},
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, use_label_encoder=False, eval_metric="logloss"
            ),
            "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        },
        "LightGBM": {
            "model": LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, verbose=-1
            ),
            "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        },
    }

    mlflow.set_experiment("telco-churn")

    results = {}
    best_model = None
    best_model_name = None
    best_accuracy = 0.0

    print("Training models...\n")
    for name, config in models.items():
        model, metrics = train_and_log(
            model=config["model"],
            model_name=name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params=config["params"],
        )
        results[name] = metrics

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_model = model
            best_model_name = name

    # Print comparison table
    print("\n" + "=" * 60)
    print("📊 Model Comparison")
    print("=" * 60)
    print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 72)
    for name, metrics in results.items():
        marker = " ⭐" if name == best_model_name else ""
        print(f"{name:<22} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} {metrics['roc_auc']:>10.4f}{marker}")

    # Save best model
    print(f"\n🏆 Best Model: {best_model_name} (accuracy={best_accuracy:.4f})")
    joblib.dump(best_model, "model.pkl")
    print("💾 Saved best model to model.pkl")

    # Save metrics report
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("📄 Saved metrics report to reports/metrics.json")

    print("\n✅ All models trained and logged to MLflow.")
    print("   Run 'mlflow ui' to explore experiment results.\n")


if __name__ == "__main__":
    main()
