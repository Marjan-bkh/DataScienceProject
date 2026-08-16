import joblib
model = joblib.load("ModelsOutcome/trip_adviser_kmeans_clusters.pkl")
print(model)
features = joblib.load("ModelsOutcome/trip_adviser_feature_columns.pkl")
print(features)

encoder = joblib.load("ModelsOutcome/trip_adviser_scaler_clusters.pkl")
print(encoder)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Datasets/tripadvisor_review.csv')

df_features = df.drop(columns=['User ID', 'Category 7'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)
df_scaled = pd.DataFrame(scaled_data, columns=df_features.columns)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# inertia_list = []
# silhouette_list = []
# k_range = range(2, 11)
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(df_scaled)
#     inertia_list.append(kmeans.inertia_)
#     silhouette_list.append(silhouette_score(df_scaled, labels))

# for k, sil in zip(k_range, silhouette_list):
#     print(f"k={k}: silhouette={sil:.4f}")

kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(df_scaled)
df['Cluster'] = cluster_labels


# cluster_profile = df.groupby('Cluster')[df_features.columns.tolist()].mean()
# print(cluster_profile.round(2))
#
# cluster_profile_viz = cluster_profile.T
# cluster_profile_viz.plot(kind='bar', figsize=(14, 6))

# outlier_cluster = df[df['Cluster'] == 1]
# print(outlier_cluster[df_features.columns].describe())
# print(outlier_cluster[['Category 4']].sort_values('Category 4', ascending=False))

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

linked = linkage(df_scaled, method='ward')
plt.figure(figsize=(14, 6))
dendrogram(linked, truncate_mode='lastp', p=30)

import joblib

joblib.dump(kmeans_final, 'ModelsOutcome/trip_adviser_kmeans_clusters.pkl')
joblib.dump(scaler, 'ModelsOutcome/trip_adviser_scaler_clusters.pkl')

feature_columns = df_features.columns.tolist()
joblib.dump(feature_columns, 'ModelsOutcome/trip_adviser_feature_columns.pkl')

print("model saved")