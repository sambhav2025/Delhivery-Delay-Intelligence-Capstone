import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def run_part_d():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 1. Load Data
    df = pd.read_csv("data/modelling_table.csv")
    
    # 2. Select numeric features for operational segmentation
    # We dynamically select numeric features (like distance, weight, cost, delivery_hours)
    exclude_cols = ["is_delayed", "shipment_id", "hub_id"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cluster_cols = [c for c in numeric_cols if c not in exclude_cols and not c.endswith("_id")]
    
    # Limit to top 5 numerical columns to keep the profile interpretable
    cluster_cols = cluster_cols[:5]
    
    # Drop NaNs just for the clustering subset
    df_cluster = df.dropna(subset=cluster_cols).copy()
    X_cluster = df_cluster[cluster_cols]
    
    # 3. Scale and Cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Using 3 clusters for High, Medium, and Low risk/effort segments
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster["segment_cluster"] = kmeans.fit_predict(X_scaled)
    
    # 4. Profile the clusters
    profile_cols = cluster_cols + (["is_delayed"] if "is_delayed" in df.columns else [])
    cluster_profile = df_cluster.groupby("segment_cluster")[profile_cols].mean().reset_index()
    
    # Count sizes
    sizes = df_cluster.groupby("segment_cluster").size().reset_index(name="shipment_count")
    cluster_profile = cluster_profile.merge(sizes, on="segment_cluster")
    
    cluster_profile.to_csv("outputs/part_d_cluster_profiles.csv", index=False)
    
    # 5. Export Models (Packaging for production)
    joblib.dump(kmeans, "models/kmeans_segmentation.pkl")
    joblib.dump(scaler, "models/cluster_scaler.pkl")
    
    # 6. Generate Operations Playbook
    playbook_content = "# Operations Playbook & Hub Segmentation\n\n"
    playbook_content += "Automated strategic recommendations based on K-Means operational clustering.\n\n"
    
    for idx, row in cluster_profile.iterrows():
        cluster_id = int(row["segment_cluster"])
        count = int(row["shipment_count"])
        
        delay_rate = row.get("is_delayed", 0) * 100
        delivery_hrs = row.get("delivery_hours", 0)
        
        playbook_content += f"## Segment {cluster_id} (Volume: {count} shipments)\n"
        playbook_content += f"- **Profile**: Average Delivery Time: {delivery_hrs:.1f} hours | Historical Delay Rate: {delay_rate:.1f}%\n"
        
        if delay_rate > 20:
            playbook_content += "- **Risk Level**: HIGH\n"
            playbook_content += "- **Action Plan**: This segment is highly prone to missing promises. Route immediately to premium carriers, bypass standard sorting queues, and trigger proactive customer SLA alerts.\n"
        elif delivery_hrs > df_cluster.get("delivery_hours", pd.Series([0])).mean():
            playbook_content += "- **Risk Level**: MEDIUM (Long Haul)\n"
            playbook_content += "- **Action Plan**: Standard delay risk but high transit times. Monitor weather and regional hub bottlenecks. Optimize load-balancing across transit nodes.\n"
        else:
            playbook_content += "- **Risk Level**: LOW\n"
            playbook_content += "- **Action Plan**: Highly stable and fast segment. Optimize strictly for margin and cost efficiency. Consolidate shipments and use standard ground routing.\n"
        playbook_content += "\n"
        
    with open("outputs/operations_playbook.md", "w") as f:
        f.write(playbook_content)
        
    print("Part D execution complete. Models packaged in models/.")
    print("Check outputs/operations_playbook.md for the final strategy report.")

if __name__ == "__main__":
    # Ensure joblib is installed
    try:
        import joblib
    except ImportError:
        print("Installing joblib...")
        os.system("pip install joblib")
        
    run_part_d()
