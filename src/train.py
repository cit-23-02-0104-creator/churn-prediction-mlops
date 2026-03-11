import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import os  
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_models():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    mlflow.set_experiment("Churn_Prediction")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, name)
            
            if name == "Random_Forest":
                joblib.dump(model, "models/model.pkl")
                print(f"Best Model ({name}) saved to models/model.pkl")

if __name__ == "__main__":
    train_models()