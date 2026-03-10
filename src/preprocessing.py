import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data():
    # 1. Path Setup - Fixed the space and added the correct folder path
    input_path = 'data/raw/dataset.csv'
    output_dir = 'data/processed'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if file exists to avoid FileNotFoundError
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found! Please check your data/raw folder.")
        return

    # 2. Data Loading
    df = pd.read_csv(input_path)
    print("Data loaded successfully...")

    # 3. Handle Missing Values & Convert TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(" ", "0")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    # 4. Encode Categorical Variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'customerID':
            df[col] = le.fit_transform(df[col])

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # 5. Train-Test Split
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Saving Processed Data for DVC tracking
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("Success: Data Engineering (Part 1) Completed!")

if __name__ == "__main__":
    preprocess_data()