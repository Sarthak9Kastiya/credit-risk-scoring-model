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
            'loan_grade_D', 'loan_grade_E', 'loan_grade_F', 'loan_grade_G'
        ]
        np.random.seed(42)
        n_samples = 1000
        X = pd.DataFrame(columns=features)
        X['customer_age'] = np.random.normal(35, 10, n_samples)
        X['customer_income'] = np.random.normal(60000, 20000, n_samples)
        X['employment_duration'] = np.random.normal(5, 3, n_samples)
        X['loan_amnt'] = np.random.normal(10000, 5000, n_samples)
        X['loan_int_rate'] = np.random.normal(11, 3, n_samples)
        X['term_years'] = np.random.choice([3, 5], n_samples)
        X['cred_hist_length'] = np.random.normal(5, 3, n_samples)
        
        for col in features[7:]:
            X[col] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
            
        # Create a mock y based on features to make model predictive and use all inputs
        # mock y based on realistic weights
        logits = -1.0 \
            + (35 - X['customer_age']) * 0.03 \
            + (60000 - X['customer_income']) / 20000 * 0.5 \
            + (5 - X['employment_duration']) * 0.1 \
            + (X['loan_amnt'] - 10000) / 5000 * 0.6 \
            + (X['loan_int_rate'] - 10.0) / 5 * 1.0 \
            + (X['term_years'] - 3) * 0.15 \
            + (5 - X['cred_hist_length']) * 0.1 \
            + X['home_ownership_RENT'] * 0.4 \
            - X['home_ownership_OWN'] * 0.4 \
            + X['home_ownership_OTHER'] * 0.2 \
            + X['loan_intent_VENTURE'] * 0.5 \
            + X['loan_intent_PERSONAL'] * 0.3 \
            + X['loan_intent_MEDICAL'] * 0.2 \
            - X['loan_intent_EDUCATION'] * 0.1 \
            - X['loan_intent_HOMEIMPROVEMENT'] * 0.2 \
            + X['loan_grade_B'] * 0.2 + X['loan_grade_C'] * 0.4 + X['loan_grade_D'] * 0.6 \
            + X['loan_grade_E'] * 0.8 + X['loan_grade_F'] * 1.0 + X['loan_grade_G'] * 1.2

        probs = 1 / (1 + np.exp(-logits))
        y = (np.random.rand(n_samples) < probs).astype(int)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Add noise and fit with strong regularization to keep probabilities soft/realistic
        model = LogisticRegression(C=0.05, class_weight='balanced')
        model.fit(X_scaled, y)
        
        joblib.dump(model, 'model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(features, 'features.joblib')
        print("Mock artifacts generated successfully. All features utilized.")

if __name__ == '__main__':
    main()
