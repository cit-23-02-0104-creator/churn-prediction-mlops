import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_log_model(model_name, model, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        # Training
        model.fit(X_train, y_train.values.ravel())
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        
        # Log Parameters and Metrics to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)
        
        print(f"{model_name} logged successfully with Accuracy: {metrics['accuracy']:.4f}")

def run_experiments():
    # Load data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    # Define Models
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    # Set MLflow experiment name
    mlflow.set_experiment("Churn_Prediction_Experiments")

    for name, model in models.items():
        train_and_log_model(name, model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    run_experiments()