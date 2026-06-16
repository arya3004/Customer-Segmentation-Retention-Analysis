import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """Engineers subscription-related features for RFM and Churn analysis."""
    df_engineered = df.copy()
    
    # 1. Frequency Proxy: Count the number of active services
    # Potential service columns
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies'
    ]
    
    # Count how many of these services are explicitly 'Yes'
    df_engineered['NumServices'] = (df_engineered[service_cols] == 'Yes').sum(axis=1)
    
    # Add InternetService if it is not 'No'
    if 'InternetService' in df_engineered.columns:
        df_engineered['NumServices'] += (df_engineered['InternetService'] != 'No').astype(int)
    
    # 2. Recency / Relationship Quality:
    # Tenure group mapping for visual segmentation and features
    def get_tenure_group(tenure):
        if tenure <= 12:
            return '0-1 Year'
        elif tenure <= 24:
            return '1-2 Years'
        elif tenure <= 48:
            return '2-4 Years'
        else:
            return '4+ Years'
            
    df_engineered['TenureGroup'] = df_engineered['tenure'].apply(get_tenure_group)
    
    # 3. Monthly charges per service (efficiency check)
    # Handle division by zero for customers with 0 services
    df_engineered['ChargePerService'] = df_engineered['MonthlyCharges'] / (df_engineered['NumServices'] + 0.1)
    
    return df_engineered

def preprocess_and_encode(df):
    """Encodes categorical variables and splits features into numeric and engineered columns."""
    df_encoded = engineer_features(df)
    
    # Drop customerID if exists
    if 'customerID' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['customerID'])
        
    # Drop TenureGroup as it is categorical and only used for visualizations/segmentation
    if 'TenureGroup' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['TenureGroup'])
    
    # List of binary features to map to 0/1
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    
    for col in binary_cols:
        if col in df_encoded.columns:
            if col == 'gender':
                df_encoded[col] = df_encoded[col].map({'Male': 1, 'Female': 0})
            else:
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
                
    # Get dummies for remaining categorical features
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"One-hot encoding categorical variables: {categorical_cols}")
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
        
    # Convert all columns to numeric just in case there are boolean columns from get_dummies
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
            
    return df_encoded

def scale_features(X_train, X_test, numeric_cols):
    """Applies standard scaling to the specified numeric features."""
    scaler = StandardScaler()
    
    # Create copies
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit and transform
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    from src.data_loader import load_and_clean_data
    df = load_and_clean_data()
    df_eng = engineer_features(df)
    print(df_eng[['tenure', 'MonthlyCharges', 'NumServices', 'ChargePerService']].head())
    df_enc = preprocess_and_encode(df)
    print("Encoded features shape:", df_enc.shape)
