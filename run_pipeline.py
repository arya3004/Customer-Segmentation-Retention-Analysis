import os
import pandas as pd
from src.data_loader import load_and_clean_data
from src.segmentation import perform_segmentation
from src.models import train_and_evaluate_models

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "segmented_customers.csv")

def main():
    print("==================================================")
    print("   STARTING CUSTOMER SEGMENTATION & CHURN PIPELINE")
    print("==================================================")
    
    # 1. Load and clean raw customer churn data
    print("\n--- Step 1: Loading & Cleaning Raw Data ---")
    df = load_and_clean_data()
    
    # 2. Run Segmentation & Customer Persona assignment
    print("\n--- Step 2: Running Customer Segmentation (Clustering) ---")
    df_segmented = perform_segmentation(df)
    
    # Save the segmented customer dataset
    df_segmented.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Segmented customer dataset saved to: {PROCESSED_DATA_PATH}")
    
    # 3. Train Churn Prediction Models
    print("\n--- Step 3: Training Churn Prediction Models ---")
    results, feature_cols = train_and_evaluate_models(df)
    
    print("\n==================================================")
    print("          PIPELINE EXECUTED SUCCESSFULLY")
    print("==================================================")

if __name__ == "__main__":
    main()
