import os
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.features import engineer_features

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def perform_segmentation(df, n_clusters=4):
    """Performs K-Means clustering on RFM-style features and assigns personas.
    
    RFM Dimensions used:
    - Recency/Loyalty proxy: 'tenure' (months with company)
    - Frequency/Breadth proxy: 'NumServices' (number of active services)
    - Monetary proxy: 'MonthlyCharges' (monthly spend)
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    # Ensure engineered features are present
    if 'NumServices' not in df.columns:
        df_feats = engineer_features(df)
    else:
        df_feats = df.copy()
        
    cluster_features = ['tenure', 'NumServices', 'MonthlyCharges']
    X_cluster = df_feats[cluster_features]
    
    # Scale features for K-Means (K-Means is sensitive to magnitude)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    df_feats['Cluster'] = cluster_labels
    
    # Determine the centroids and map to business personas dynamically
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    persona_map = {}
    if n_clusters == 4:
        tenures = centroids[:, 0]
        charges = centroids[:, 2]
        
        # Sort indices by tenure
        sorted_by_tenure = np.argsort(tenures)
        
        # 1. At-Risk Casuals: lowest mean tenure (very new customers)
        at_risk_idx = sorted_by_tenure[0]
        
        # Out of the remaining 3 clusters, sort by monthly charges
        remaining_indices = list(sorted_by_tenure[1:])
        charges_remaining = [charges[idx] for idx in remaining_indices]
        
        # 2. High-Value VIPs: highest charges among the remaining
        vip_idx = remaining_indices[np.argmax(charges_remaining)]
        
        # 3. Budget Loyalists: lowest charges among the remaining
        remaining_indices.remove(vip_idx)
        charges_remaining_2 = [charges[idx] for idx in remaining_indices]
        budget_idx = remaining_indices[np.argmin(charges_remaining_2)]
        
        # 4. Standard/Mid-tier: the last one
        remaining_indices.remove(budget_idx)
        standard_idx = remaining_indices[0]
        
        persona_map = {
            at_risk_idx: "At-Risk Casuals",
            vip_idx: "High-Value VIPs",
            budget_idx: "Budget Loyalists",
            standard_idx: "Standard/Mid-Tier"
        }
    else:
        for i in range(n_clusters):
            persona_map[i] = f"Segment {i}"
            
    df_feats['CustomerPersona'] = df_feats['Cluster'].map(persona_map)
    
    # Save the clustering artifacts
    with open(os.path.join(MODEL_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(MODEL_DIR, "segmentation_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "persona_map.pkl"), "wb") as f:
        pickle.dump(persona_map, f)
        
    print("Segmentation completed. Personas mapped:")
    for cluster_id, name in persona_map.items():
        centroid = centroids[cluster_id]
        print(f" - {name} (Cluster {cluster_id}): Tenure={centroid[0]:.1f}m, Services={centroid[1]:.1f}, MonthlySpend=${centroid[2]:.2f}")
        
    return df_feats

if __name__ == "__main__":
    from src.data_loader import load_and_clean_data
    df = load_and_clean_data()
    df_segmented = perform_segmentation(df)
    print(df_segmented['CustomerPersona'].value_counts())
