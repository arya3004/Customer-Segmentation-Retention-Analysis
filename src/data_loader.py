import os
import urllib.request
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")

def download_dataset():
    """Downloads the Telco Customer Churn dataset if it does not exist locally."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}")
        
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        try:
            urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
            print(f"Dataset successfully downloaded and saved to {RAW_DATA_PATH}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise e
    else:
        print(f"Dataset already exists at {RAW_DATA_PATH}")

def load_and_clean_data():
    """Loads the dataset and performs initial preprocessing and cleaning.
    
    Cleaning steps:
    1. Coerces 'TotalCharges' to numeric and fills empty string values (associated with tenure=0) with 0.0.
    2. Converts target 'Churn' to binary integers (1 = Yes, 0 = No).
    """
    # Ensure data is downloaded
    download_dataset()
    
    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Clean 'TotalCharges' - replace empty strings or spaces with NaN, then fill with 0
    # (Customers with tenure=0 haven't been charged for a full month yet, so TotalCharges should be 0)
    df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
    null_count = df['TotalCharges'].isnull().sum()
    if null_count > 0:
        print(f"Found {null_count} rows with blank TotalCharges (tenure=0). Filling with 0.0...")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges']).fillna(0.0)
    else:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
        
    # Map 'Churn' to binary (1 = Yes, 0 = No)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        print("Mapped Churn targets to 1 (Yes) and 0 (No).")
        
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.info())
    print(df.head(2))
