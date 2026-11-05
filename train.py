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
    dataset_path = '/Users/sarthakkastiya/Desktop/ITSP/LoanDataset - LoansDatasest.csv'
    
    if os.path.exists(dataset_path):
        print("Loading data...")
        df = pd.read_csv(dataset_path)

        print("Preprocessing data...")
        df['customer_id'] = df['customer_id'].apply(to_int)
        df = df.dropna(subset=['customer_id'])
        
        median_age = df[df['customer_age'] <= 90]['customer_age'].median()
        df.loc[df['customer_age'] > 90, 'customer_age'] = median_age
        
        df['employment_duration'] = df['employment_duration'].fillna(df['employment_duration'].median())
        df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())
        
        df = df.drop('historical_default', axis='columns')
        
        df = df.dropna(subset=['loan_amnt', 'Current_loan_status'])
        df['Current_loan_status'] = df['Current_loan_status'].replace({'DEFAULT': 1, 'NO DEFAULT': 0})
        df['customer_income'] = df['customer_income'].apply(to_int)
        df = df.dropna()
        df['loan_amnt'] = df['loan_amnt'].apply(str_to_float)

        dummies = pd.get_dummies(df[['home_ownership','loan_intent','loan_grade']], drop_first=False).astype(int)
        dummies = dummies.drop(['home_ownership_OTHER','loan_intent_DEBTCONSOLIDATION','loan_grade_A'], axis=1)
        
        df1 = df.drop(['customer_id','home_ownership','loan_intent','loan_grade'], axis='columns')
        df1 = pd.concat([df1, dummies], axis=1)

        y = df1['Current_loan_status'].astype(int)
        X = df1.drop(['Current_loan_status'], axis=1)

        print("Columns in X:", list(X.columns))

        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        print("Training model...")
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train_scaled, y_train)

        print("Saving artifacts...")
        joblib.dump(model, 'model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        print("Done!")
    else:
        print("Error: Dataset not found!")

if __name__ == '__main__':
    main()
