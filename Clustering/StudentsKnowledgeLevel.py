import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
df = pd.read_excel("Datasets/Students_Knowledge_Level.xlsx")

# print(df.shape)
# print(df.dtypes)
# print(df.head())
# print(df.describe())

features = ['V1', 'V2', 'V3', 'V4', 'V5']
print(df.groupby('Class')[features].mean().round(3))

# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()
# for i, col in enumerate(features):
#     axes[i].hist(df[col], bins=20, color='steelblue', edgecolor='black')
#     axes[i].set_xlabel(col)
# axes[5].axis('off')
# plt.tight_layout()
# plt.show()
#

# plt.figure(figsize=(10, 6))
# df[features].boxplot()
# plt.show()

# plt.figure(figsize=(8, 5))
# df['Class'].value_counts().sort_index().plot(kind='bar', color='coral', edgecolor='black')
# plt.show()

# plt.figure(figsize=(8, 6))
# corr_matrix = df[features].corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
# plt.tight_layout()
# plt.show()

# print(df.isnull().sum())
# print(df.duplicated().sum())

X = df[features].copy()
y_true = df['Class'].copy()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)
# print(X.describe().loc[['mean', 'std']])
# print(X_scaled.describe().loc[['mean', 'std']])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# k_range = range(2, 11)
# inertia_values = []  # برای Elbow Method
# silhouette_values = []  # برای Silhouette Score
#
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
#     cluster_labels = kmeans.fit_predict(X_scaled)
#
#     inertia_values.append(kmeans.inertia_)
#     sil_score = silhouette_score(X_scaled, cluster_labels)
#     silhouette_values.append(sil_score)
#     print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={sil_score:.4f}")

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# axes[0].plot(k_range, inertia_values, marker='o', color='steelblue')
# axes[0].set_xlabel('تعداد خوشه‌ها (k)')
# axes[0].set_ylabel('Inertia')
# axes[0].set_title('Elbow Method')
# axes[0].grid(True)
#
# axes[1].plot(k_range, silhouette_values, marker='o', color='coral')
# axes[1].set_xlabel('تعداد خوشه‌ها (k)')
# axes[1].set_ylabel('Silhouette Score')
# axes[1].set_title('Silhouette Score برای مقادیر مختلف k')
# axes[1].grid(True)
# plt.tight_layout()
# plt.show()

#  K-Means   k=5
kmeans_final = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
cluster_labels_kmeans = kmeans_final.fit_predict(X_scaled)

df['KMeans_Cluster'] = cluster_labels_kmeans
# print(pd.Series(cluster_labels_kmeans).value_counts().sort_index())
#
# # --- کاهش بُعد با PCA برای مصورسازی ---
# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=2, random_state=42)
# X_pca = pca.fit_transform(X_scaled)
#
# print(f" {pca.explained_variance_ratio_}")
# print(f" {pca.explained_variance_ratio_.sum():.2%}")
#

# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#
# scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels_kmeans,
#                              cmap='viridis', alpha=0.6, s=40)
# axes[0].set_xlabel('PC1')
# axes[0].set_ylabel('PC2')
# axes[0].set_title('خوشه‌های K-Means (k=5) در فضای PCA')
# plt.colorbar(scatter1, ax=axes[0], label='خوشه')
#
# scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=df['Class'],
#                              cmap='viridis', alpha=0.6, s=40)
# axes[1].set_xlabel('PC1')
# axes[1].set_ylabel('PC2')
# axes[1].set_title('Class واقعی در فضای PCA (برای مقایسه)')
# plt.colorbar(scatter2, ax=axes[1], label='Class')
# plt.tight_layout()
# plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

linkage_matrix = linkage(X_scaled, method='ward')

# plt.figure(figsize=(14, 6))
# dendrogram(linkage_matrix, truncate_mode='lastp', p=30,
#            leaf_rotation=90, leaf_font_size=10)
# plt.title('Dendrogram (Hierarchical Clustering با روش Ward)')
# plt.xlabel('نمونه‌ها (یا خوشه‌های ادغام‌شده)')
# plt.ylabel('فاصله (Ward Distance)')
# plt.axhline(y=15, color='red', linestyle='--', label='برش پیشنهادی برای k=5')
# plt.legend()
# plt.tight_layout()
# plt.show()

hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
cluster_labels_hier = hierarchical.fit_predict(X_scaled)

df['Hierarchical_Cluster'] = cluster_labels_hier
# print("(Hierarchical):")
# print(pd.Series(cluster_labels_hier).value_counts().sort_index())
sil_hier = silhouette_score(X_scaled, cluster_labels_hier)
# print(f"\nSilhouette Score برای Hierarchical (k=5): {sil_hier:.4f}")
# print(f"Silhouette Score برای K-Means (k=5) بود: {silhouette_values[3]:.4f}")  # k=5 چهارمین عنصره چون از k=2 شروع کردیم

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=5, random_state=42, n_init=10)
cluster_labels_gmm = gmm.fit_predict(X_scaled)

df['GMM_Cluster'] = cluster_labels_gmm
# print(pd.Series(cluster_labels_gmm).value_counts().sort_index())

sil_gmm = silhouette_score(X_scaled, cluster_labels_gmm)
print(f"\nSilhouette Score برای GMM (k=5): {sil_gmm:.4f}")
# print(f"K-Means:      {silhouette_values[3]:.4f}")
# print(f"Hierarchical: {sil_hier:.4f}")
# print(f"GMM:          {sil_gmm:.4f}")

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
import numpy as np

y_true = df['Class'].values

results = {}
for name, labels in [('K-Means', cluster_labels_kmeans),
                      ('Hierarchical', cluster_labels_hier),
                      ('GMM', cluster_labels_gmm)]:
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    results[name] = {'ARI': ari, 'NMI': nmi}
    print(f"{name}:")
    print(f"  ARI (Adjusted Rand Index): {ari:.4f}")
    print(f"  NMI (Normalized Mutual Information): {nmi:.4f}\n")

# best_algo = max(results, key=lambda x: results[x]['ARI'])
# print(f" ARI: {best_algo}")
#
# label_map = {'K-Means': cluster_labels_kmeans,
#              'Hierarchical': cluster_labels_hier,
#              'GMM': cluster_labels_gmm}
#
# crosstab = pd.crosstab(df['Class'], label_map[best_algo],
#                         rownames=['Class واقعی'], colnames=['خوشه‌ی پیش‌بینی‌شده'])
# print(f"\nجدول تطبیق (Class واقعی در مقابل خوشه‌های {best_algo}):")
# print(crosstab)

crosstab = pd.crosstab(df['Class'], cluster_labels_gmm, rownames=['True Class'], colnames=['GMM Cluster'])
# print(crosstab)


cluster_profile = df.groupby('GMM_Cluster')[features].mean()
cluster_profile['count'] = df.groupby('GMM_Cluster').size()
# print(cluster_profile.round(3))

# print(cluster_profile.round(3))

# plt.figure(figsize=(10, 6))
# sns.heatmap(cluster_profile[features], annot=True, cmap='YlOrRd', fmt='.2f',
#             cbar_kws={'label': 'مقدار میانگین (Scale نشده)'})
# plt.title('پروفایل خوشه‌های GMM بر اساس میانگین فیچرها')
# plt.xlabel('فیچر')
# plt.ylabel('خوشه')
# plt.tight_layout()
# plt.show()
#
# import joblib
#
# joblib.dump(gmm, 'ModelsOutcome/students_knowledge_gmm_final_model.pkl')
# joblib.dump(scaler, 'ModelsOutcome/students_knowledge_scaler.pkl')
#
# print("model saved")
