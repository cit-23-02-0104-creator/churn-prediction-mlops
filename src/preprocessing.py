import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data():
    input_path = 'data/raw/dataset.csv'
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(input_path)
    
    # Cleaning
    df['TotalCharges'] = df['TotalCharges'].replace(" ", "0")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    # Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'customerID':
            df[col] = le.fit_transform(df[col])
    
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling & Saving Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl") 

    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    print(" Preprocessing complete. Scaler saved to models/scaler.pkl")

if __name__ == "__main__":
    preprocess_data()