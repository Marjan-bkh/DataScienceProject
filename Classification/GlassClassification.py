import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
df = pd.read_csv('Datasets/glass.data', header=None, names=columns)
df = df.drop('Id', axis=1)
# print(df.shape)
# print(df.head())
# print(df.info())
# print(df['Type'].value_counts().sort_index())
# print(df.describe())
#
# df.drop('Type', axis=1).hist(figsize=(12, 10), bins=20, edgecolor='black')
# plt.tight_layout()
# plt.show()
#
# features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
# fig, axes = plt.subplots(3, 3, figsize=(15, 12))
# axes = axes.flatten()
#
# for i, feature in enumerate(features):
#     sns.boxplot(x='Type', y=feature, data=df, ax=axes[i])
#     axes[i].set_title(f'{feature} by Glass Type')
# plt.tight_layout()
# plt.show()

# corr_matrix = df.drop('Type', axis=1).corr()
# print(corr_matrix.round(2))
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
# plt.title('Correlation Matrix')
# plt.tight_layout()
# plt.show()
#
# features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
# for col in features:
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#     outliers = df[(df[col] < lower) | (df[col] > upper)]
#     print(f"{col}: {len(outliers)} outlier ({len(outliers)/len(df)*100:.1f}%)")
#Add features
df['Ca_Na_ratio'] = df['Ca'] / df['Na']
df['Al_Si_ratio'] = df['Al'] / df['Si']
df['Total_modifiers'] = df['Na'] + df['Mg'] + df['Ca'] + df['K']
# binary features for zero-inflated
df['has_Ba'] = (df['Ba'] > 0).astype(int)
df['has_Fe'] = (df['Fe'] > 0).astype(int)
# print(df.head())
# print(df.shape)
# print(df[['has_Ba', 'has_Fe']].sum())

from sklearn.model_selection import train_test_split
X = df.drop('Type', axis=1)
y = df['Type']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# print("Train shape:", X_train.shape)
# print("Test shape:", X_test.shape)
# #classes distribution
# print(y_train.value_counts(normalize=True).sort_index().round(3))
# print(y_test.value_counts(normalize=True).sort_index().round(3))
# print(y_test.value_counts().sort_index())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
# print(X_train_scaled.describe().round(2))

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, rf_pred))
# print(classification_report(y_test, rf_pred,zero_division=0))

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
# print("Accuracy:", accuracy_score(y_test, knn_pred))
# print(classification_report(y_test, knn_pred,zero_division=0))

from sklearn.model_selection import StratifiedKFold, cross_val_score

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='f1_macro')
# print("Scores:", rf_cv_scores.round(3))
# print(f"Mean: {rf_cv_scores.mean():.3f}, Std: {rf_cv_scores.std():.3f}")
#
# knn_cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=skf, scoring='f1_macro')
# print("Scores:", knn_cv_scores.round(3))
# print(f"Mean: {knn_cv_scores.mean():.3f}, Std: {knn_cv_scores.std():.3f}")

importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)
# print(importances.round(3))
#
# plt.figure(figsize=(10, 6))
# importances.plot(kind='barh')
# plt.gca().invert_yaxis()
# plt.xlabel('Importance')
# plt.title('Random Forest - Feature Importance')
# plt.tight_layout()
# plt.show()

import joblib

joblib.dump(rf_model, 'ModelsOutcome/glass_type_rf_model.pkl')
joblib.dump(list(X_train.columns), 'ModelsOutcome/glass_model_features.pkl')

print("مدل با موفقیت ذخیره شد.")
print("فیچرهای مورد نیاز مدل:", list(X_train.columns))

loaded_model = joblib.load('ModelsOutcome/glass_type_rf_model.pkl')
test_pred = loaded_model.predict(X_test)
print(accuracy_score(y_test, test_pred))