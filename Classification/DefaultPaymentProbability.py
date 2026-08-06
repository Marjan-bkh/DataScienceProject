import pandas as pd
import numpy as np

df= pd.read_excel('Datasets/default of credit card clients.xls',header=1)
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.duplicated().sum())
# print(df.isnull().sum().sum())
# print(df['EDUCATION'].value_counts())
# print(df['MARRIAGE'].value_counts())
# for col in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
#     print(col, sorted(df[col].unique()))
# print(df['default payment next month'].value_counts())
# print(df['default payment next month'].value_counts(normalize=True))

df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
# print(df['EDUCATION'].value_counts())
# print(df['MARRIAGE'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# sns.barplot(x='SEX', y='default payment next month', data=df, ax=axes[0])
# axes[0].set_title('Default Rate by SEX')
# sns.barplot(x='EDUCATION', y='default payment next month', data=df, ax=axes[1])
# axes[1].set_title('Default Rate by EDUCATION')
# sns.barplot(x='MARRIAGE', y='default payment next month', data=df, ax=axes[2])
# axes[2].set_title('Default Rate by MARRIAGE')
# plt.tight_layout()
# plt.show()
#
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# sns.boxplot(x='default payment next month', y='LIMIT_BAL', data=df, ax=axes[0])
# sns.boxplot(x='default payment next month', y='AGE', data=df, ax=axes[1])
# plt.tight_layout()
# plt.show()
#
# bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
# plt.figure(figsize=(6, 5))
# sns.heatmap(df[bill_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
# plt.title('Correlation between BILL_AMT columns')
# plt.show()

bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
df['bill_trend'] = df['BILL_AMT1'] - df['BILL_AMT6']

pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
df['avg_pay_amt'] = df[pay_amt_cols].mean(axis=1)

pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
df['avg_pay_status'] = df[pay_status_cols].mean(axis=1)
# print(df[['avg_bill_amt', 'bill_trend', 'avg_pay_amt', 'avg_pay_status']].describe())
# print(df.shape)

df = pd.get_dummies(df, columns=['EDUCATION', 'MARRIAGE'], drop_first=True)
# print(df.columns.tolist())
# print(df.shape)
# print(df.head())

bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

X = df.drop(columns=['ID', 'default payment next month'])
y = df['default payment next month']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# print(X_train.shape, X_test.shape)
# print(y_train.value_counts(normalize=True))
# print(y_test.value_counts(normalize=True))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# models = {
#     'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), X_train_scaled, X_test_scaled),
#     'Random Forest': (RandomForestClassifier(class_weight='balanced', random_state=42), X_train, X_test),
#     'Gradient Boosting': (GradientBoostingClassifier(random_state=42), X_train, X_test),
# }
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
# results_df = pd.DataFrame(results)
# print(results_df)

from sklearn.model_selection import cross_validate

# scoring = ['accuracy', 'precision', 'recall', 'f1']
# cv_results = []
# for name, (model, X_tr, X_te) in models.items():
#     scores = cross_validate(model, X_tr, y_train, cv=5, scoring=scoring)
#     cv_results.append({
#         'Model': name,
#         'Accuracy': scores['test_accuracy'].mean(),
#         'Precision': scores['test_precision'].mean(),
#         'Recall': scores['test_recall'].mean(),
#         'Recall_std': scores['test_recall'].std(),
#     })
# cv_results_df = pd.DataFrame(cv_results)
# print(cv_results_df)

final_model = RandomForestClassifier(class_weight='balanced', random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

import joblib

joblib.dump(final_model, 'ModelsOutcome/default_payment_random_forest_model.pkl')
joblib.dump(list(X_train.columns), 'ModelsOutcome/default_payment_random_forest_model_features.pkl')

print("model saved.")
