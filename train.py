import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

def to_int(x):
    try:
        return int(x)
    except:
        return None

def str_to_float(strr):
    try:
        str1 = strr[1:] 
        lst = str1.split(',')
        str_f = ''
        if len(lst) > 0:
            for i in range(len(lst)):
                str_f = str_f + lst[i]
        return float(str_f)
    except:
        return None

def main():
    dataset_path = '/Users/sarthakkastiya/Downloads/LoanDataset - LoansDatasest.csv'
    
    if os.path.exists(dataset_path):
        print("Loading data...")
        df = pd.read_csv(dataset_path)

        print("Preprocessing data...")
        df['customer_id'] = df['customer_id'].apply(to_int)
        df = df.dropna(subset=['customer_id'])
        
        median_age = df[df['customer_age'] <= 90]['customer_age'].median()
        df.loc[df['customer_age'] > 90, 'customer_age'] = median_age
        
        df['employment_duration'] = df['employment_duration'].fillna(df['employment_duration'].median())
        df = df.dropna(subset=['loan_amnt', 'Current_loan_status'])
        df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())
        df['Current_loan_status'] = df['Current_loan_status'].replace({'DEFAULT': 1, 'NO DEFAULT': 0})
        df['customer_income'] = df['customer_income'].apply(to_int)
        df = df.dropna()
        df['loan_amnt'] = df['loan_amnt'].apply(str_to_float)

        dummies = pd.get_dummies(df[['home_ownership','loan_intent','loan_grade']], drop_first=True).astype(int)
        df = df.drop(['home_ownership','loan_intent','loan_grade'], axis=1)
        df = pd.concat([df, dummies], axis=1)

        y = df['Current_loan_status']
        X = df.drop(['Current_loan_status', 'customer_id'], axis=1)

        feature_names = list(X.columns)

        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("Training model...")
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train_scaled, y_train)

        score = model.score(X_test_scaled, y_test)
        print(f"Model accuracy: {score:.4f}")

        print("Saving artifacts...")
        joblib.dump(model, 'model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(feature_names, 'features.joblib')
        print("Done!")
    else:
        print("Dataset not found! Creating a mock model so the app can still run.")
        features = [
            'customer_age', 'customer_income', 'employment_duration', 'loan_amnt',
            'loan_int_rate', 'term_years', 'cred_hist_length',
            'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
            'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
            'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_grade_B', 'loan_grade_C',
            'loan_grade_D', 'loan_grade_E'
        ]
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, len(features)), columns=features)
        y = np.random.randint(0, 2, size=100)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression()
        model.fit(X_scaled, y)
        
        joblib.dump(model, 'model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(features, 'features.joblib')
        print("Mock artifacts generated successfully. (Please place the CSV in Downloads to train on real data)")

if __name__ == '__main__':
    main()
