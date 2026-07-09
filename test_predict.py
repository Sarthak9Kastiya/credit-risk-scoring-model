data = {
    "customer_age": 30,
    "customer_income": 50000.0,
    "employment_duration": 5.0,
    "loan_amnt": 10000.0,
    "loan_int_rate": 10.0,
    "term_years": 5,
    "cred_hist_length": 5,
    "home_ownership": "OWN",
    "loan_intent": "PERSONAL",
    "loan_grade": "A"
}

try:
    import pandas as pd
    import joblib
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, 'model.joblib'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))
    features = joblib.load(os.path.join(BASE_DIR, 'features.joblib'))
    
    df = pd.DataFrame([data])
    dummies = pd.get_dummies(df[['home_ownership','loan_intent','loan_grade']])
    df = df.drop(['home_ownership','loan_intent','loan_grade'], axis=1)
    df = pd.concat([df, dummies], axis=1)
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]
    print(df)
    X_scaled = scaler.transform(df)
    print("X_scaled:", X_scaled)
    prob = model.predict_proba(X_scaled)
    print("Prob:", prob)
except Exception as e:
    print("Error:", e)
