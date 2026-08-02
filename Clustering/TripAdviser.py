import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Datasets/tripadvisor_review.csv')
# print(df.shape)
# print(df.head())
# print(df.dtypes)
# print(df.info())
# print(df.describe().to_string())
# print(df.describe().T)
# print(df.isnull().sum())
# print(df.duplicated().sum())

# df.iloc[:, 1:].hist(bins=20, figsize=(14, 10))
# plt.tight_layout()
# plt.show()
#
# corr = df.iloc[:, 1:].corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
# plt.show()

# print(df.isnull().sum())
# print(df.isnull().sum().sum())
# df.iloc[:, 1:].boxplot(figsize=(14, 6), rot=45)
# plt.tight_layout()
# plt.show()

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
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# axes[0].plot(k_range, inertia_list, marker='o')
# axes[0].set_xlabel('تعداد خوشه (k)')
# axes[0].set_ylabel('Inertia')
# axes[0].set_title('Elbow Method')
# axes[1].plot(k_range, silhouette_list, marker='o', color='orange')
# axes[1].set_xlabel('تعداد خوشه (k)')
# axes[1].set_ylabel('Silhouette Score')
# axes[1].set_title('Silhouette Analysis')
# plt.tight_layout()
# plt.show()
# for k, sil in zip(k_range, silhouette_list):
#     print(f"k={k}: silhouette={sil:.4f}")

kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(df_scaled)
df['Cluster'] = cluster_labels
# print(df['Cluster'].value_counts().sort_index())
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2, random_state=42)
# pca_result = pca.fit_transform(df_scaled)
# print(f"واریانس توضیح داده شده توسط هر مولفه: {pca.explained_variance_ratio_}")
# print(f"مجموع واریانس توضیح داده شده: {pca.explained_variance_ratio_.sum():.4f}")
#
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('K-Means Clusters (k=3) - PCA Visualization')
# plt.colorbar(scatter, label='Cluster')
# plt.show()

# cluster_profile = df.groupby('Cluster')[df_features.columns.tolist()].mean()
# print(cluster_profile.round(2))
#
# cluster_profile_viz = cluster_profile.T
# cluster_profile_viz.plot(kind='bar', figsize=(14, 6))
# plt.title('میانگین امتیاز هر دسته به تفکیک خوشه')
# plt.ylabel('میانگین امتیاز')
# plt.xlabel('دسته')
# plt.legend(title='Cluster')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# outlier_cluster = df[df['Cluster'] == 1]
# print(outlier_cluster[df_features.columns].describe())
# print(outlier_cluster[['Category 4']].sort_values('Category 4', ascending=False))

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

linked = linkage(df_scaled, method='ward')
plt.figure(figsize=(14, 6))
dendrogram(linked, truncate_mode='lastp', p=30)
# plt.title('Dendrogram (Ward Linkage)')
# plt.xlabel('نمونه‌ها / خوشه‌ها')
# plt.ylabel('فاصله')
# plt.show()
# hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
# hier_labels = hierarchical.fit_predict(df_scaled)
# print(pd.Series(hier_labels).value_counts().sort_index())

import joblib

joblib.dump(kmeans_final, 'ModelsOutcome/trip_adviser_kmeans_clusters.pkl')
joblib.dump(scaler, 'ModelsOutcome/trip_adviser_scaler_clusters.pkl')

feature_columns = df_features.columns.tolist()
joblib.dump(feature_columns, 'ModelsOutcome/trip_adviser_feature_columns.pkl')

print("مدل، اسکیلر و لیست فیچرها ذخیره شدند.")