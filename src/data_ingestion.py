import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_data():
    
    url = "https://dagshub.com/cit-23-02-0104-creator/churn-prediction-mlops/raw/main/data/raw/dataset.csv"
    output_path = "data/raw/dataset.csv"
    
    
    os.makedirs("data/raw", exist_ok=True)
    
    try:
        logging.info(f"Downloading data from {url}...")
        df = pd.read_csv(url)
        
        
        df.to_csv(output_path, index=False)
        logging.info(f" Data ingestion successful. Saved to {output_path}. Shape: {df.shape}")
        
    except Exception as e:
        logging.error(f" Error during ingestion: {e}")

if __name__ == "__main__":
    ingest_data()