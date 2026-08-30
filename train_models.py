import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error

def train_and_save():
    print("Loading processed data for training...")
    os.makedirs('models', exist_ok=True)
    
    df = pd.read_csv('data/processed/master_df.csv')
    model_df = df.dropna(subset=['days_rec_to_sanc']).copy()
    
    categorical_features = ['State', 'Work category']
    numeric_features = [
        'Sanction Amount ( ₹ )', 
        'RECOMMENDED AMOUNT ( ₹ )', 
        'amount_diff',
        'days_rec_to_sanc', 
        'mp_avg_utilization', 
        'state_delay_rate'
    ]
    
    # Preprocessor using dense arrays for HistGradientBoosting compatibility
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')), 
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')), 
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])
    
    # --- 1. Train Risk Classifier ---
    print("Training Risk Classification Model...")
    X_risk = model_df[numeric_features + categorical_features]
    y_risk = model_df['is_high_risk']
    
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)
    
    risk_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(
            class_weight={0: 1.0, 1: 3.5}, 
            max_iter=200, 
            max_depth=7,
            random_state=42
        ))
    ])
    risk_pipeline.fit(X_tr_r, y_tr_r)
    
    print("\n--- Risk Model Metrics ---")
    print(classification_report(y_te_r, risk_pipeline.predict(X_te_r)))
    joblib.dump(risk_pipeline, 'models/risk_model.joblib')
    print("Risk model saved to models/risk_model.joblib")
    
    # --- 2. Train Fund Forecasting Regressor ---
    print("\nTraining Fund Forecasting Model...")
    forecast_df = model_df[model_df['total_disbursed'] > 0]
    X_f = forecast_df[numeric_features + categorical_features]
    y_f = forecast_df['fund_utilization_ratio']
    
    X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(X_f, y_f, test_size=0.2, random_state=42)
    
    fund_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.08,
            max_depth=8,
            random_state=42
        ))
    ])
    fund_pipeline.fit(X_tr_f, y_tr_f)
    
    rmse = np.sqrt(mean_squared_error(y_te_f, fund_pipeline.predict(X_te_f)))
    print(f"Fund Model RMSE: {rmse:.4f}")
    joblib.dump(fund_pipeline, 'models/fund_model.joblib')
    print("Fund forecasting model saved to models/fund_model.joblib")

if __name__ == "__main__":
    train_and_save()