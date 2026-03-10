from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Churn Prediction Service")

# 1. Load your trained model
try:
    model = joblib.load("models/model.pkl") 
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# 2. Define Input Schema (Basic inputs)
class ChurnInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def read_root():
    return {"status": "API is online", "message": "Go to /docs for testing"}

# 3. Predict Endpoint
@app.post("/predict")
def predict(data: ChurnInput):
    if model is None:
        return {"error": "Model file not found in models/ folder."}
    
    # Convert input to DataFrame
    input_dict = data.model_dump() # Pydantic v2 use model_dump instead of dict
    df = pd.DataFrame([input_dict])
    
    # --- Fix for Feature Mismatch ---
    # Get the features the model was trained on
    try:
        model_features = model.feature_names_in_
        # Add missing columns with default value 0
        for col in model_features:
            if col not in df.columns:
                df[col] = 0
        
        # Ensure columns are in the exact same order as training
        df = df[model_features]
    except AttributeError:
        # If the model doesn't have feature_names_in_, it will use the 3 columns
        pass
    # --------------------------------

    # Get Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "churn_probability": round(float(probability), 2),
        "prediction": "Yes" if prediction == 1 else "No"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)