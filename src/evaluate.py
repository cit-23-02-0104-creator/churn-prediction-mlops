import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model():
    input_dir = 'data/processed'
    model_path = 'models/model.pkl'

    # Check if model exists
    if not os.path.exists(model_path):
        print("Error: Model file not found in models/ folder!")
        return

    # Loading Test Data and Trained Model
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")
    y_test = pd.read_csv(f"{input_dir}/y_test.csv")
    model = joblib.load(model_path)

    # Making Predictions
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # Calculating Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)

    # Save to metrics.txt
    with open("metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(report)
    print("\nSuccess: Evaluation results saved to metrics.txt")


if __name__ == "__main__":
    evaluate_model()