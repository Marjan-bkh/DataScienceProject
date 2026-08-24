import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Datasets/Wholesale_customers_data.csv')
# print(df.shape)
# print(df.head())
# print(df.info())
# print(df.describe().to_string())
# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df.skew())

cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()
# for i, col in enumerate(cols):
#     axes[i].hist(df[col], bins=30, color='steelblue', edgecolor='black')
#     axes[i].set_title(col)
# plt.tight_layout()
# plt.show()
#
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()
# for i, col in enumerate(cols):
#     axes[i].boxplot(df[col])
#     axes[i].set_title(col)
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(8, 6))
# corr = df[cols].corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
# plt.show()
# print(corr)
# def count_outliers_iqr(df, col):
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
#     return len(outliers), lower_bound, upper_bound
# print(f"{'Column':<20}{'Outlier Count':<15}{'Outlier %':<12}{'Lower':<12}{'Upper'}")
# for col in cols:
#     count, low, up = count_outliers_iqr(df, col)
#     pct = (count / len(df)) * 100
#     print(f"{col:<20}{count:<15}{pct:<12.2f}{low:<12.1f}{up:.1f}")

df_log = df.copy()
for col in cols:
    df_log[col] = np.log1p(df[col])
# print(df[cols].skew())
# print(df_log[cols].skew())
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()
# for i, col in enumerate(cols):
#     axes[i].hist(df_log[col], bins=30, color='seagreen', edgecolor='black')
#     axes[i].set_title(f'{col} (log-transformed)')
# plt.tight_layout()
# plt.show()

from sklearn.preprocessing import StandardScaler
X = df_log[cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=cols)
# print(X.describe().loc[['mean', 'std']])
# print(X_scaled.describe().loc[['mean', 'std']])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# inertia_list = []
# silhouette_list = []
# k_range = range(2, 11)
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_scaled)
#     inertia_list.append(kmeans.inertia_)
#     silhouette_list.append(silhouette_score(X_scaled, labels))
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# axes[0].plot(k_range, inertia_list, marker='o')
# axes[0].set_xlabel('تعداد Cluster (k)')
# axes[0].set_ylabel('Inertia (WCSS)')
# axes[0].set_title('Elbow Method')
# axes[1].plot(k_range, silhouette_list, marker='o', color='darkorange')
# axes[1].set_xlabel('تعداد Cluster (k)')
# axes[1].set_ylabel('Silhouette Score')
# axes[1].set_title('Silhouette Score')
# plt.tight_layout()
# plt.show()
# for k, inertia, sil in zip(k_range, inertia_list, silhouette_list):
#     print(f"k={k}:  Inertia={inertia:.2f}   Silhouette={sil:.4f}")

from sklearn.cluster import KMeans

kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_2 = kmeans_2.fit_predict(X_scaled)
score = silhouette_score(X_scaled, labels_2)
print(f"silhouette score = {score:.4f}")

kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_3 = kmeans_3.fit_predict(X_scaled)

df_result = df.copy()
df_result['Cluster_k2'] = labels_2
# df_result['Cluster_k3'] = labels_3

print(df_result['Cluster_k2'].value_counts().sort_index())
# print(df_result['Cluster_k3'].value_counts().sort_index())

print(df_result.groupby('Cluster_k2')[cols].mean().round(1))
# print(df_result.groupby('Cluster_k3')[cols].mean().round(1))
#
print(pd.crosstab(df_result['Cluster_k2'], df_result['Channel']))
# print(pd.crosstab(df_result['Cluster_k3'], df_result['Channel']))

# import joblib
# joblib.dump(kmeans_2, 'ModelsOutcome/wholesale_kmeans_model.pkl')
# joblib.dump(scaler, 'ModelsOutcome/wholesale_scaler.pkl')
# joblib.dump(cols, 'ModelsOutcome/wholesale_feature_columns.pkl')
# print("model saved")
#
# loaded_model = joblib.load('ModelsOutcome/wholesale_kmeans_model.pkl')
# loaded_scaler = joblib.load('ModelsOutcome/wholesale_scaler.pkl')
