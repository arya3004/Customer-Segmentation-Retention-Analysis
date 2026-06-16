import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.features import preprocess_and_encode, scale_features

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def train_and_evaluate_models(df):
    """Trains and compares Logistic Regression, Random Forest, and XGBoost models.
    
    Selects the best model based on F1-Score (since customer churn is an imbalanced classification problem)
    and saves the model, scaler, and training columns.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    # Preprocess and encode categorical columns
    df_encoded = preprocess_and_encode(df)
    
    # Target variable 'Churn'
    y = df_encoded['Churn']
    X = df_encoded.drop(columns=['Churn'])
    
    # Save the feature column names to ensure input alignment during prediction
    feature_cols = X.columns.tolist()
    with open(os.path.join(MODEL_DIR, "model_features.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
        
    # Split train and test (stratified because of class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numeric features (tenure, MonthlyCharges, TotalCharges, ChargePerService, NumServices)
    # We want to identify the columns to scale
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'NumServices', 'ChargePerService']
    numeric_cols = [col for col in numeric_cols if col in X.columns]
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, numeric_cols)
    
    # Save the classification scaler
    with open(os.path.join(MODEL_DIR, "classification_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    # Define models
    # Set class weights/scale_pos_weight to handle class imbalance (~26% churners)
    ratio = (len(y_train) - sum(y_train)) / sum(y_train) # negative to positive ratio
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio, random_state=42)
    }
    
    results = {}
    best_f1 = 0
    best_model_name = ""
    best_model_obj = None
    
    print("\n--- Model Training & Evaluation ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predict
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "Confusion Matrix": cm,
            "model_obj": model
        }
        
        print(f" {name} Performance:")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}  (Out of predicted churners, how many actually churn?)")
        print(f"   Recall:    {rec:.4f}  (Out of actual churners, how many did we catch?)")
        print(f"   F1-Score:  {f1:.4f}  (Harmonic mean of precision and recall)")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        # Track best model based on F1-Score (balances Precision and Recall for imbalanced data)
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model
            
    print(f"\nBest Model Selected: {best_model_name} with F1-Score of {best_f1:.4f}")
    
    # Save the best model
    with open(os.path.join(MODEL_DIR, "best_churn_model.pkl"), "wb") as f:
        pickle.dump(best_model_obj, f)
        
    # Save metadata about which model was chosen
    with open(os.path.join(MODEL_DIR, "model_metadata.pkl"), "wb") as f:
        pickle.dump({"model_name": best_model_name, "metrics": results[best_model_name]}, f)
        
    # Save all results for display on the dashboard
    clean_results = {k: {m: v for m, v in metrics.items() if m != 'model_obj'} for k, metrics in results.items()}
    with open(os.path.join(MODEL_DIR, "all_model_results.pkl"), "wb") as f:
        pickle.dump(clean_results, f)
        
    return results, feature_cols

def predict_single_customer(customer_data_dict):
    """Predicts the churn probability for a single customer.
    
    customer_data_dict should contain the raw keys of the customer data.
    """
    # Load model, scaler, and features
    with open(os.path.join(MODEL_DIR, "best_churn_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "classification_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model_features.pkl"), "rb") as f:
        model_features = pickle.load(f)
        
    # Make a DataFrame from the dictionary
    df_raw = pd.DataFrame([customer_data_dict])
    
    # Engineer the features (NumServices, ChargePerService)
    # 1. Frequency Proxy: count the number of services
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies'
    ]
    df_raw['NumServices'] = (df_raw[service_cols] == 'Yes').sum(axis=1)
    if 'InternetService' in df_raw.columns:
        df_raw['NumServices'] += (df_raw['InternetService'] != 'No').astype(int)
        
    # 2. Charge per service
    df_raw['ChargePerService'] = df_raw['MonthlyCharges'] / (df_raw['NumServices'] + 0.1)
    
    # Map binary features
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df_raw.columns:
            if col == 'gender':
                df_raw[col] = df_raw[col].map({'Male': 1, 'Female': 0})
            else:
                df_raw[col] = df_raw[col].map({'Yes': 1, 'No': 0})
                
    # One-hot encode using get_dummies
    df_encoded = pd.get_dummies(df_raw)
    
    # Reindex columns to match model features exactly, filling missing columns with 0
    df_encoded = df_encoded.reindex(columns=model_features, fill_value=0)
    
    # Scale numeric columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'NumServices', 'ChargePerService']
    numeric_cols = [col for col in numeric_cols if col in model_features]
    df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])
    
    # Predict probability
    prob = model.predict_proba(df_encoded)[0, 1]
    prediction = int(model.predict(df_encoded)[0])
    
    return prediction, prob

if __name__ == "__main__":
    from src.data_loader import load_and_clean_data
    df = load_and_clean_data()
    results, feat_cols = train_and_evaluate_models(df)
