import pandas as pd
import numpy as np

column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
df = pd.read_csv('Datasets/data_banknote_authentication.txt', header=None, names=column_names)
# print(df.shape)
# print(df.info())
# print(df.head())
# print(df['class'].value_counts())
# print(df['class'].value_counts(normalize=True))
# print(df.describe())
# print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# features = ['variance', 'skewness', 'curtosis', 'entropy']
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# axes =axes.flatten()
# for i, feature in enumerate(features):
#     sns.histplot(data=df, x=feature, hue='class', kde=True, ax=axes[i], bins=30)
#     axes[i].set_title(f'Distribution of {feature} by class')
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix')
# plt.show()
# print(df.corr()['class'].sort_values())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.value_counts(normalize=True))
# print(y_test.value_counts(normalize=True))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
# print(X_train_scaled.describe())

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    # 'Logistic Regression': (LogisticRegression(random_state=42), X_train_scaled, X_test_scaled),
    # 'KNN': (KNeighborsClassifier(), X_train_scaled, X_test_scaled),
    'SVM': (SVC(random_state=42), X_train_scaled, X_test_scaled),
    # 'Decision Tree': (DecisionTreeClassifier(random_state=42), X_train, X_test),
    # 'Random Forest': (RandomForestClassifier(random_state=42), X_train, X_test),
    # 'Gradient Boosting': (GradientBoostingClassifier(random_state=42), X_train, X_test),
}

# results = []
# for name, (model, X_tr, X_te) in models.items():
#     model.fit(X_tr, y_train)
#     y_pred = model.predict(X_te)
#     results.append({
#         'Model': name,
#         'Accuracy': accuracy_score(y_test, y_pred),
#         'Precision': precision_score(y_test, y_pred),
#         'Recall': recall_score(y_test, y_pred),
#         'F1': f1_score(y_test, y_pred)
#     })
# results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
# print(results_df.to_string(index=False))

from sklearn.model_selection import cross_val_score, StratifiedKFold

# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_results = []
# for name, (model, X_tr, X_te) in models.items():
#     scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='f1')
#     cv_results.append({
#         'Model': name,
#         'CV F1 Mean': scores.mean(),
#         'CV F1 Std': scores.std(),
#         'Stability (mean-std)': scores.mean() - scores.std()
#     })
# cv_df = pd.DataFrame(cv_results).sort_values('Stability (mean-std)', ascending=False)
# print(cv_df.to_string(index=False))


final_model = SVC(random_state=42)
final_model.fit(X_train_scaled, y_train)

y_pred_final = final_model.predict(X_test_scaled)

# print("Accuracy:", accuracy_score(y_test, y_pred_final))
# print("Precision:", precision_score(y_test, y_pred_final))
# print("Recall:", recall_score(y_test, y_pred_final))
# print("F1:", f1_score(y_test, y_pred_final))
#
# from sklearn.metrics import confusion_matrix, classification_report
# print(confusion_matrix(y_test, y_pred_final))
# print(classification_report(y_test, y_pred_final))

import joblib

joblib.dump(final_model, 'ModelsOutcome/banknote_svm_model.pkl')
joblib.dump(scaler, 'ModelsOutcome/banknote_scaler.pkl')
joblib.dump(list(X_train.columns), 'ModelsOutcome/banknote_feature_columns.pkl')

print("model saved")


