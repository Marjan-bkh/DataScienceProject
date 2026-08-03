import pandas as pd
import numpy as np

df = pd.read_csv(
    'Datasets/household_power_consumption.txt',
    sep=';',
    na_values=['?'],
    low_memory=False
)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.drop(columns=['Date', 'Time'], inplace=True)
# print(df.shape)
# print(df.dtypes)
# print(df.head())
# print(df.isnull().sum())
# print(df.memory_usage(deep=True).sum() / 1024**2, "MB")

# missing_mask = df['Global_active_power'].isnull()
# # print(df.loc[missing_mask, 'Datetime'].min())
# # print(df.loc[missing_mask, 'Datetime'].max())
#
# missing_dates = df.loc[missing_mask, 'Datetime'].dt.date
# # print(missing_dates.value_counts().sort_index().head(20))
# # print("تعداد روزهای متفاوتی که حداقل یک missing دارن:", missing_dates.nunique())
# missing_by_day = missing_dates.value_counts().sort_values(ascending=False)
# print(missing_by_day.head(15))
# print(missing_by_day[missing_by_day > 1000])

df = df.sort_values('Datetime').reset_index(drop=True)
cols_to_fill = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
df[cols_to_fill] = df[cols_to_fill].interpolate(method='linear', limit=60, limit_direction='forward')
# print(df[cols_to_fill].isnull().sum())

df = df.set_index('Datetime')
daily = df.resample('D').agg({
    'Global_active_power': ['mean', 'max', 'std', 'sum'],
    'Global_reactive_power': 'mean',
    'Voltage': 'mean',
    'Global_intensity': 'mean',
    'Sub_metering_1': 'sum',
    'Sub_metering_2': 'sum',
    'Sub_metering_3': 'sum'
})
daily.columns = ['_'.join(col) for col in daily.columns]
missing_ratio = df['Global_active_power'].isnull().resample('D').mean()
daily['missing_ratio'] = missing_ratio
# print(daily.shape)
# print(daily.head())
# print(daily['missing_ratio'].describe())

threshold = 0.1  # اگه بیش از 10% دقایق یک روز گمشده باشه، اون روز حذف میشه
daily_clean = daily[daily['missing_ratio'] <= threshold].copy()
daily_clean = daily_clean.drop(columns=['missing_ratio'])

import matplotlib.pyplot as plt
import seaborn as sns

# fig, ax = plt.subplots(figsize=(15, 4))
# daily_clean['Global_active_power_mean'].plot(ax=ax)
# ax.set_title('روند میانگین مصرف روزانه در طول زمان')
# plt.tight_layout()
# plt.show()
#
# daily_clean.hist(figsize=(15, 10), bins=30)
# plt.tight_layout()
# plt.show()
#
# corr = daily_clean.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
# plt.tight_layout()
# plt.show()
# print(corr['Global_active_power_mean'].sort_values(ascending=False))

features = daily_clean.copy()
# (redundant)
features = features.drop(columns=['Global_intensity_mean', 'Voltage_mean',
                                    'Global_active_power_sum'])
#Add some features
total_sub_metering = (features['Sub_metering_1_sum'] +
                       features['Sub_metering_2_sum'] +
                       features['Sub_metering_3_sum'])

features['Sub_metering_1_ratio'] = features['Sub_metering_1_sum'] / total_sub_metering
features['Sub_metering_2_ratio'] = features['Sub_metering_2_sum'] / total_sub_metering
features['Sub_metering_3_ratio'] = features['Sub_metering_3_sum'] / total_sub_metering
features[['Sub_metering_1_ratio', 'Sub_metering_2_ratio', 'Sub_metering_3_ratio']] = \
    features[['Sub_metering_1_ratio', 'Sub_metering_2_ratio', 'Sub_metering_3_ratio']].fillna(0)
#. فیچر نوسان نسبی (coefficient of variation) به‌جای std خام
features['Global_active_power_cv'] = (
    features['Global_active_power_std'] / features['Global_active_power_mean']
)
#delete unnecessary
features = features.drop(columns=['Sub_metering_1_sum', 'Sub_metering_2_sum',
                                    'Sub_metering_3_sum', 'Global_active_power_std'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# inertia_list = []
# silhouette_list = []
# k_range = range(2, 11)
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(features_scaled)
#     inertia_list.append(kmeans.inertia_)
#     silhouette_list.append(silhouette_score(features_scaled, labels))
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# axes[0].plot(list(k_range), inertia_list, marker='o')
# axes[0].set_title('Elbow Method')
# axes[0].set_xlabel('k')
# axes[0].set_ylabel('Inertia')
#
# axes[1].plot(list(k_range), silhouette_list, marker='o', color='orange')
# axes[1].set_title('Silhouette Score')
# axes[1].set_xlabel('k')
# axes[1].set_ylabel('Silhouette Score')
# plt.tight_layout()
# plt.show()
# for k, sil in zip(k_range, silhouette_list):
#     print(f"k={k}: silhouette={sil:.4f}")

kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(features_scaled)
features['KMeans_Cluster'] = kmeans_labels
# print(features['KMeans_Cluster'].value_counts().sort_index())
# print(silhouette_score(features_scaled, kmeans_labels))

# from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.cluster import AgglomerativeClustering
#
# linkage_matrix = linkage(features_scaled, method='ward')
# plt.figure(figsize=(15, 6))
# dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
# plt.title('Dendrogram (Ward Linkage)')
# plt.xlabel('Cluster Size')
# plt.ylabel('Distance')
# plt.tight_layout()
# plt.show()
#
# hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
# hierarchical_labels = hierarchical.fit_predict(features_scaled)
# features['Hierarchical_Cluster'] = hierarchical_labels
# print(features['Hierarchical_Cluster'].value_counts().sort_index())
# print(silhouette_score(features_scaled, hierarchical_labels))

# cluster_profile = features.groupby('KMeans_Cluster')[
#     ['Global_active_power_mean', 'Global_active_power_max',
#      'Global_reactive_power_mean', 'Sub_metering_1_ratio',
#      'Sub_metering_2_ratio', 'Sub_metering_3_ratio',
#      'Global_active_power_cv']
# ].mean()
# print(cluster_profile.round(3).to_string())
# print("days count")
# print(features['KMeans_Cluster'].value_counts().sort_index())

import joblib
feature_columns = ['Global_active_power_mean', 'Global_active_power_max',
                    'Global_reactive_power_mean', 'Sub_metering_1_ratio',
                    'Sub_metering_2_ratio', 'Sub_metering_3_ratio',
                    'Global_active_power_cv']

joblib.dump(kmeans_final, 'ModelsOutcome/power_consumption_kmeans_model.pkl')
joblib.dump(scaler, 'ModelsOutcome/power_consumption_scaler.pkl')
joblib.dump(feature_columns, 'ModelsOutcome/power_consumption_feature_columns.pkl')
features.to_csv('Datasets/power_consumption_daily_with_clusters.csv')
print("ذخیره شد.")
