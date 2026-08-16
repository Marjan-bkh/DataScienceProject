import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

columns = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']
df = pd.read_csv('Datasets/bupa.data', header=None, names=columns)
df = df.drop(columns=['selector'])
# print(df.shape)
# print(df.head())
# print(df.info())
# print(df.describe().to_string())
features = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']
# fig, ax = plt.subplots(2,3,figsize=(15,10))
# axes = ax.flatten()
# for i, col in enumerate(features):
#     axes[i].hist(df[col], bins=10, color='steelblue', edgecolor='black')
#     axes[i].set_xlabel(col)
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(10,6))
# df[features].boxplot()
# plt.show()
#
# plt.figure(figsize=(10,6))
# corr_matrix = df[features].corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
# plt.show()

# from scipy.stats import skew
# for col in df.columns:
#     print(f"{col}: skewness = {skew(df[col]):.2f}")

import numpy as np
df_log = df.copy()
cols_to_log = ['sgpt', 'sgot', 'gammagt','alkphos', 'drinks']
for col in cols_to_log:
    df_log[col] = np.log1p(df_log[col])
# print(df_log.describe())
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# for i, col in enumerate(cols_to_log):
#     axes[0, i].hist(df[col], bins=20, color='steelblue')
#     axes[0, i].set_title(f'{col} - Original')
#     axes[1, i].hist(df_log[col], bins=20, color='darkorange')
#     axes[1, i].set_title(f'{col} - Log Transformed')
# plt.tight_layout()
# plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_log)
df_scaled = pd.DataFrame(df_scaled, columns=df_log.columns)
# print(df_scaled.describe())

from sklearn.cluster import KMeans

inertia_values = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia_values.append(kmeans.inertia_)

# plt.figure(figsize=(8, 5))
# plt.plot(k_range, inertia_values, marker='o')
# plt.xlabel('تعداد Cluster (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')
# plt.xticks(k_range)
# plt.grid(True)
# plt.show()

from sklearn.metrics import silhouette_score

silhouette_scores = []
k_range_sil = range(2, 11)

# for k in k_range_sil:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(df_scaled)
#     score = silhouette_score(df_scaled, labels)
#     silhouette_scores.append(score)
#     print(f"k={k}: silhouette score = {score:.4f}")
#
# plt.figure(figsize=(8, 5))
# plt.plot(k_range_sil, silhouette_scores, marker='o', color='green')
# plt.xlabel('تعداد Cluster (k)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Analysis')
# plt.xticks(k_range_sil)
# plt.grid(True)
# plt.show()

kmeans_k2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_k2 = kmeans_k2.fit_predict(df_scaled)
df['cluster_k2'] = labels_k2
print(df['cluster_k2'].value_counts())
print(df.groupby('cluster_k2')[['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']].mean().round(2))


# print(scaler.feature_names_in_)
# import joblib
#
# joblib.dump(kmeans_k2, 'ModelsOutcome/liver_disorder_kmeans_k2_model.pkl')
#
# joblib.dump(scaler, 'ModelsOutcome/liver_disorder_scaler.pkl')
#
# joblib.dump(cols_to_log, 'ModelsOutcome/liver_disorder_cols_to_log.pkl')
# print("model saved")