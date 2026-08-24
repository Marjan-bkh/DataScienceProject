import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

columns = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']
df = pd.read_csv('Datasets/bupa.data', header=None, names=columns)
df = df.drop(columns=['selector'])

features = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']

import numpy as np
df_log = df.copy()
cols_to_log = ['sgpt', 'sgot', 'gammagt','alkphos', 'drinks']
for col in cols_to_log:
    df_log[col] = np.log1p(df_log[col])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_log)
df_scaled = pd.DataFrame(df_scaled, columns=df_log.columns)

from sklearn.cluster import KMeans

inertia_values = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia_values.append(kmeans.inertia_)

from sklearn.metrics import silhouette_score

silhouette_scores = []
k_range_sil = range(2, 11)

kmeans_k2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_k2 = kmeans_k2.fit_predict(df_scaled)
df['cluster_k2'] = labels_k2
score = silhouette_score(df_scaled, labels_k2)
print(f"silhouette score = {score:.4f}")
print(df['cluster_k2'].value_counts())
print(df.groupby('cluster_k2')[['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']].mean().round(2))

