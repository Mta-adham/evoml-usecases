#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:31:08 2025

@author: victor.villar
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# For optional 3D plotting
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
# 1. Load synthetic data (adjust the path as needed)
###############################################################################
df_synthetic = pd.read_csv(
    "/home/manal/Workspace/evoml-usecases/src/synthetic_credit_data.csv"
)

###############################################################################
# 2. (Optionally) drop the binary targets if you only want pure unsupervised
#    segmentation. If you plan to interpret them, you can keep them.
###############################################################################
df_for_clustering = df_synthetic.drop(
    columns=["HighIncome", "HighBalance"], 
    errors="ignore"
)

# Identify categorical and numeric columns. Adjust names to match your data.
cat_cols = ["Gender", "Student", "Married", "Ethnicity"]
numeric_cols = [col for col in df_for_clustering.columns if col not in cat_cols]

###############################################################################
# 3. One-hot encode the categorical columns
###############################################################################
df_encoded = pd.get_dummies(df_for_clustering, columns=cat_cols, drop_first=True)

###############################################################################
# 4. Scale all features for K-means
###############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

###############################################################################
# 5. Run K-means
###############################################################################
k = 5  # Example: 5 clusters
kmeans = KMeans(n_clusters=k, random_state=92)
kmeans.fit(X_scaled)

###############################################################################
# 6. Get cluster labels. Optionally assign risk labels
###############################################################################
cluster_labels = kmeans.labels_

risk_mapping = {
    0: "Low Risk", 
    1: "Moderate Risk", 
    2: "Medium Risk", 
    3: "High Risk", 
    4: "Very High Risk"
}
risk_labels = [risk_mapping[label] for label in cluster_labels]

# Attach cluster info to both your scaled DataFrame and your original DataFrame
df_encoded["Cluster"] = cluster_labels
df_encoded["RiskLabel"] = risk_labels

df_for_clustering["Cluster"] = cluster_labels
df_for_clustering["RiskLabel"] = risk_labels


###############################################################################
# 7. Examine "Centroids" in the Original (Unscaled) Space
###############################################################################
# Means (or medians) for numeric columns by cluster.
cluster_means = (
    df_for_clustering
    .groupby("Cluster")[numeric_cols]
    .mean()
    # Rename index from 0..4 to risk labels
    .rename(index=risk_mapping)
)

print("=== Cluster Means (Unscaled) ===")
print(cluster_means)
print()

# Check how many people per cluster.
cluster_sizes = df_for_clustering["Cluster"].value_counts()

# Replace numeric indices with risk labels
cluster_sizes.index = cluster_sizes.index.map(risk_mapping)
print("=== Cluster Sizes ===")
print(cluster_sizes)
print()

###############################################################################
# 8. Visualize Clusters in 2D Using PCA
###############################################################################
pca_2d = PCA(n_components=2, random_state=92)
X_pca_2d = pca_2d.fit_transform(X_scaled)  # Project data onto 2 principal components

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=cluster_labels,
    cmap="viridis",
    alpha=0.7
)

# Create a legend mapping cluster label to color
legend_labels = list(risk_mapping.values())
for i, lbl in enumerate(legend_labels):
    plt.scatter([], [], c=scatter.cmap(scatter.norm(i)), label=lbl)

plt.legend(
    scatterpoints=1, 
    frameon=True, 
    labelspacing=1, 
    title="Risk Segment"
)
plt.title("K-Means Clusters (PCA 2D Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

###############################################################################
# 9. (Optional) 3D PCA Visualization
###############################################################################
pca_3d = PCA(n_components=3, random_state=92)
X_pca_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter_3d = ax.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=cluster_labels,
    cmap="viridis",
    alpha=0.7
)

# 3D legend
for i, lbl in enumerate(legend_labels):
    ax.scatter([], [], [], c=scatter_3d.cmap(scatter_3d.norm(i)), label=lbl)

ax.legend(
    scatterpoints=1,
    frameon=True,
    labelspacing=1,
    title="Risk Segment"
)

ax.set_title("K-Means Clusters (PCA 3D Projection)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.show()
