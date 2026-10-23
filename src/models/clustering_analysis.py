"""
WattWise AI - Load Profile Clustering Analysis
Implements:
1. Daily Load Profile Feature Representation
2. K-Means Clustering with Silhouette Score and Davies-Bouldin Index
3. Agglomerative Hierarchical Clustering with Ward Linkage and Dendrogram
4. Cluster Characterization and Peak Pattern Discovery
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

def run_clustering_experiments(
    hourly_path: str = "data/processed/clean_hourly_national.csv",
    output_dir: str = "report_artifacts",
    figures_dir: str = "results/figures"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Loading hourly data for profile clustering from {hourly_path}...")
    df = pd.read_csv(hourly_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    # Pivot to Daily Profiles: rows = dates, cols = hours 0..23
    daily_matrix = df.pivot(index="date", columns="hour", values="demand_mw")
    # Drop any incomplete days (should be full 24h)
    daily_matrix = daily_matrix.dropna(how="any")
    print(f"Formed {len(daily_matrix)} complete 24-hour daily demand vectors.")

    # 1. Feature Representation:
    # Standardized Daily Profile: Normalized by mean and std to capture diurnal shape variation
    scaler = StandardScaler()
    X_shape = scaler.fit_transform(daily_matrix.values)
    
    # 2. Evaluate K-Means across k in [2, 3, 4, 5, 6]
    clustering_results = []
    k_range = [2, 3, 4, 5, 6]
    kmeans_models = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_shape)
        sil = silhouette_score(X_shape, labels)
        db = davies_bouldin_score(X_shape, labels)
        kmeans_models[k] = (km, labels)
        
        clustering_results.append({
            "Algorithm": "K-Means",
            "Feature Representation": "24-Hour Standardized Diurnal Shape",
            "Cluster Count": k,
            "Silhouette Score": round(sil, 4),
            "Davies-Bouldin Index": round(db, 4),
            "Cluster Interpretation": f"Partitions daily load into {k} diurnal profile regimes",
            "Limitations": "Assumes spherical clusters; sensitive to initialization"
        })

    # 3. Evaluate Agglomerative Hierarchical Clustering across k in [2, 3, 4, 5, 6]
    hierarchical_models = {}
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        labels = agg.fit_predict(X_shape)
        sil = silhouette_score(X_shape, labels)
        db = davies_bouldin_score(X_shape, labels)
        hierarchical_models[k] = (agg, labels)
        
        clustering_results.append({
            "Algorithm": "Hierarchical (Ward)",
            "Feature Representation": "24-Hour Standardized Diurnal Shape",
            "Cluster Count": k,
            "Silhouette Score": round(sil, 4),
            "Davies-Bouldin Index": round(db, 4),
            "Cluster Interpretation": f"Dendrogram-based hierarchical taxonomy into {k} profile regimes",
            "Limitations": "Computationally quadratic with number of days; non-probabilistic"
        })

    cluster_df = pd.DataFrame(clustering_results)
    print("\n================== DEMAND PROFILE CLUSTERING COMPARISON ==================")
    print(cluster_df.to_string(index=False))

    cluster_df.to_csv(os.path.join(output_dir, "clustering_model_comparison.csv"), index=False)

    # 4. Deep Analysis on Optimal Cluster Count (k=3)
    best_k = 3
    best_km, best_labels = kmeans_models[best_k]
    daily_matrix["cluster"] = best_labels

    # Compute actual demand centroids (in MW) for each cluster
    cluster_centroids_mw = daily_matrix.groupby("cluster").mean()

    # Map cluster characteristics
    # Cluster interpretations in India:
    # High Summer Afternoon/Evening Spike vs Winter Lighting vs Moderate Monsoon Base
    cluster_profiles_info = []
    regime_names = ["Moderate Flat / Monsoon Profile", "Summer High-Peak Afternoon Diurnal", "Winter / Spring Evening Lighting Peak"]
    
    # Sort clusters by average demand level to assign intuitive names
    cluster_means = cluster_centroids_mw.mean(axis=1).sort_values()
    sorted_cluster_ids = cluster_means.index.tolist()

    name_mapping = {
        sorted_cluster_ids[0]: "Low / Moderate Base Load (Monsoon / Mild Season)",
        sorted_cluster_ids[1]: "Intermediate Dual-Peak Load (Spring / Autumn)",
        sorted_cluster_ids[2]: "Extreme High Peak Load (Summer Heatwave & Pre-Monsoon)"
    }

    for c_id in range(best_k):
        cnt = int((best_labels == c_id).sum())
        centroid_vals = cluster_centroids_mw.loc[c_id].values.tolist()
        assigned_name = name_mapping.get(c_id, f"Profile Regime {c_id}")
        cluster_profiles_info.append({
            "cluster_id": c_id,
            "regime_name": assigned_name,
            "day_count": cnt,
            "percentage": round(cnt / len(daily_matrix) * 100.0, 1),
            "mean_demand_mw": round(float(np.mean(centroid_vals)), 1),
            "peak_demand_mw": round(float(np.max(centroid_vals)), 1),
            "peak_hour": int(np.argmax(centroid_vals)),
            "min_demand_mw": round(float(np.min(centroid_vals)), 1),
            "min_hour": int(np.argmin(centroid_vals)),
            "hourly_profile_mw": [round(v, 1) for v in centroid_vals]
        })

    with open(os.path.join(output_dir, "clustering_profiles.json"), "w") as f:
        json.dump(cluster_profiles_info, f, indent=2)

    # 5. Generate Visualizations:
    # A. Cluster Diurnal Profile Curves
    plt.figure(figsize=(10, 6))
    hours = list(range(24))
    colors = ["#0284c7", "#10b981", "#ef4444"]
    for i, prof in enumerate(cluster_profiles_info):
        plt.plot(
            hours, prof["hourly_profile_mw"],
            label=f"{prof['regime_name']} ({prof['percentage']}%)",
            color=colors[i % len(colors)],
            linewidth=2.5
        )
    plt.title("WattWise AI — Daily Electricity Demand Profiles (K-Means k=3)", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day (0-23)", fontsize=11)
    plt.ylabel("Demand (MW)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.xticks(hours)
    plt.tight_layout()
    centroid_fig_path = os.path.join(figures_dir, "cluster_daily_profiles.png")
    plt.savefig(centroid_fig_path, dpi=200)
    plt.close()
    print(f"Saved cluster profiles figure to {centroid_fig_path}")

    # B. Hierarchical Clustering Dendrogram
    plt.figure(figsize=(12, 6))
    # Sample 80 days evenly for clear dendrogram rendering
    step = max(1, len(X_shape) // 80)
    sample_indices = np.arange(0, len(X_shape), step)
    Z = linkage(X_shape[sample_indices], method="ward")
    dendrogram(Z, p=30, truncate_mode="level", leaf_rotation=90, leaf_font_size=9)
    plt.title("WattWise AI — Agglomerative Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=13, fontweight="bold")
    plt.xlabel("Sampled Daily Observation Clusters", fontsize=11)
    plt.ylabel("Ward Linkage Distance", fontsize=11)
    plt.tight_layout()
    dendro_fig_path = os.path.join(figures_dir, "hierarchical_dendrogram.png")
    plt.savefig(dendro_fig_path, dpi=200)
    plt.close()
    print(f"Saved dendrogram figure to {dendro_fig_path}")

    return cluster_df

if __name__ == "__main__":
    run_clustering_experiments()
