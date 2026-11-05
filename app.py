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
    print("Successfully loaded model artifacts.")
except Exception as e:
    print(f"Warning: Could not load model artifacts: {e}")
    model, scaler = None, None

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
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
        
    data = application.dict()
    x = np.zeros(19)
    
    cs = data.get('credit_score', 700)
    if cs > 900:
        return {"error": "Invalid Credit Score"}
        
    x[0] = data['customer_age']
    x[1] = data['customer_income']
    x[2] = data['employment_duration']
    x[3] = data['loan_amnt']
    
    base_rate = 6.0
    if 750 <= cs <= 900:
        loan_int_rate = base_rate + 1.5
    elif 600 <= cs < 750:
        loan_int_rate = base_rate + 5.0
    elif cs < 600:
        loan_int_rate = base_rate + 10.0
        
    x[4] = loan_int_rate
    x[5] = data['term_years']
    x[6] = data['cred_hist_length']
    
    home_ownership = data.get('home_ownership', '')
    if home_ownership == 'MORTGAGE':
        x[7] = 1
    if home_ownership == 'OWN':
        x[8] = 1
    if home_ownership == 'RENT':
        x[9] = 1
        
    loan_intent = data.get('loan_intent', '')
    if loan_intent == 'EDUCATION':
        x[10] = 1
    if loan_intent == 'HOMEIMPROVEMENT':
        x[11] = 1
    if loan_intent == 'MEDICAL':
        x[12] = 1
    if loan_intent == 'PERSONAL':
        x[13] = 1
    if loan_intent == 'VENTURE':
        x[14] = 1
        
    loan_grade = 'A'
    if 650 <= cs <= 750:
        x[15] = 1
        loan_grade = 'B'
    elif 550 <= cs < 650:
        x[16] = 1
        loan_grade = 'C'
    elif 300 <= cs < 550:
        x[17] = 1
        loan_grade = 'D'
    elif cs < 300:
        x[18] = 1
        loan_grade = 'E'
        
    x_scaled = scaler.transform([x])
    prob = model.predict_proba(x_scaled)[0][1]
    pred = 1 if prob > 0.4 else 0
    
    return {
        "prediction": int(pred),
        "status": "DEFAULT" if pred == 1 else "NO DEFAULT",
        "probability_of_default": float(prob),
        "interest_rate": loan_int_rate,
        "loan_grade": loan_grade
    }

# Mount static files for frontend
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
