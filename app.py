from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import uvicorn

app = FastAPI(title="Credit Risk Scoring Model for Banks")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load artifacts
try:
    model = joblib.load(os.path.join(BASE_DIR, 'model.joblib'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))
    features = joblib.load(os.path.join(BASE_DIR, 'features.joblib'))
    print("Successfully loaded model artifacts.")
except Exception as e:
    print(f"Warning: Could not load model artifacts: {e}")
    model, scaler, features = None, None, None

class LoanApplication(BaseModel):
    customer_age: int
    customer_income: float
    employment_duration: float
    loan_amnt: float
    credit_score: int
    term_years: int
    cred_hist_length: int
    home_ownership: str
    loan_intent: str

@app.post("/api/predict")
def predict(application: LoanApplication):
    if not model or not scaler or not features:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
        
    data = application.dict()
    df = pd.DataFrame([data])
    
    # Process categorical variables as done in training
    dummies = pd.get_dummies(df[['home_ownership','loan_intent']])
    df = df.drop(['home_ownership','loan_intent'], axis=1)
    df = pd.concat([df, dummies], axis=1)
    
    # Ensure all columns from training are present, and in the correct order
    for col in features:
        if col not in df.columns:
            df[col] = 0
            
    df = df[features] # reorder to match exact feature set
    
    # Scale
    X_scaled = scaler.transform(df)
    
    # Predict
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]
    
    return {
        "prediction": int(pred),
        "status": "DEFAULT" if pred == 1 else "NO DEFAULT",
        "probability_of_default": float(prob)
    }

# Mount static files for frontend
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
