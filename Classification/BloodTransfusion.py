import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


columns =[ 'Recency','Frequency','Monetary','Time','Target']
df = pd.read_csv('Datasets/transfusion.data',header=0, names=columns)
# اطمینان از عددی بودن همه‌ی ستون‌ها
# skipinitialspace موقع خوندن فایل هم فاصله‌های اضافه رو مدیریت می‌کنه
# df = df.apply(pd.to_numeric)
# print(df.head().to_string())
# print(df.info())
# print(df.describe().to_string())
# print(df.shape)
# print(df.isnull().sum())
# print(df.columns.values)
# print(df['Target'].value_counts())
features = ['Recency', 'Frequency', 'Monetary', 'Time']
# fig, axes = plt.subplots(2,2, figsize=(14,10))
# axes =axes.flatten()
# for i,col in enumerate(features):
#     sns.histplot(data= df, x=col, hue= 'Target', ax=axes[i],kde=True,multiple='stack')
#     axes[i].set_title(f'{col} by Target')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,10))
# corr = df.corr()
# sns.heatmap(corr, annot=True, cmap='YlGnBu',fmt='.2f',center=0)
# plt.tight_layout()
# plt.show()
#
# fig, ax = plt.subplots(2,2)
# ax=ax.flatten()
# for i,col in enumerate(features):
#     sns.boxplot(x='Target', y=col, data=df, ax=ax[i])
# plt.tight_layout()
# plt.show()

# def detect_outliers_iqr(data,column):
#     Q1 = data[column].quantile(0.25)
#     Q3 = data[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = data[(data[column]< lower_bound )|(data[column] > upper_bound)]
#     return outliers,lower_bound,upper_bound
# for column in features:
#     outliers,lb,ub = detect_outliers_iqr(df,column)
#     print(f"{column}: تعداد outlier = {len(outliers)}  |  محدوده مجاز = [{lb:.2f}, {ub:.2f}]")

from sklearn.model_selection import train_test_split
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42,stratify=y)
# print(X_train.shape,X_test.shape)
# print (y_train.value_counts(normalize=True))
# print (y_test.value_counts(normalize=True))
X_train = X_train.copy()
X_test = X_test.copy()
X_train['Frequency_per_Time'] = X_train['Frequency']/X_train['Time'] # نسبت Frequency به Time
X_test['Frequency_per_Time'] = X_test['Frequency']/X_test['Time']
# print(X_train[['Frequency','Time','Frequency_per_Time']].head())
X_train = X_train.drop('Monetary', axis=1)
X_test = X_test.drop('Monetary', axis=1)

X_train_log = X_train.copy()
X_test_log = X_test.copy()

X_train_log['Frequency'] = np.log1p(X_train['Frequency'])
X_test_log['Frequency'] = np.log1p(X_test['Frequency'])

X_train_log['Recency'] = np.log1p(X_train['Recency'])
X_test_log['Recency'] = np.log1p(X_test['Recency'])
# print(X_train_log.head())
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# sns.histplot(X_train['Frequency'], kde=True, ax=axes[0])
# axes[0].set_title('Frequency - Before Log')
# sns.histplot(X_train_log['Frequency'], kde=True, ax=axes[1])
# axes[1].set_title('Frequency - After Log')
# plt.tight_layout()
# plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
}
# results = []
# for name, model in models.items():
#     if name == 'Logistic Regression':
#         X_tr, X_te = X_train_log, X_test_log
#     else:
#         X_tr, X_te = X_train, X_test
#
#     model.fit(X_tr, y_train)
#     y_pred = model.predict(X_te)
#     y_pred_proba = model.predict_proba(X_te)[:, 1]  # احتمال کلاس 1
#     results.append({
#         'Model': name,
#         'Accuracy': accuracy_score(y_test, y_pred),
#         'Precision': precision_score(y_test, y_pred),
#         'Recall': recall_score(y_test, y_pred),
#         'F1-Score': f1_score(y_test, y_pred),
#         'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
#     })
# results_df = pd.DataFrame(results)
# print(results_df.to_string(index=False))

from sklearn.model_selection import cross_val_score, StratifiedKFold
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_results =[]
# for name, model in models.items():
#     X_data = X_train_log if name == 'Logistic Regression' else X_train
#     scores = cross_val_score(model, X_data, y_train, cv=cv, scoring='f1')
#     cv_results.append({
#         'Model': name,
#         'Mean F1': scores.mean(),
#         'Std F1': scores.std()
#     })
# results_df = pd.DataFrame(cv_results)
# print(results_df.to_string(index=False))

final_model = RandomForestClassifier(random_state=42, n_estimators=100)
final_model.fit(X_train, y_train)
y_pred_final = final_model.predict(X_test)
y_proba_final = final_model.predict_proba(X_test)[:, 1]
#
print("--- Random Forest - نتیجه‌ی نهایی روی Test ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_final):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_final):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_final):.3f}")
print(f"F1-score:  {f1_score(y_test, y_pred_final):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_final):.3f}")

# importances = final_model.feature_importances_
# feature_names = X_train.columns
# importances_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances}).sort_values(by='Importance', ascending=False)
# print(importances_df.to_string(index=False))
#
# import joblib
# joblib.dump(final_model, 'ModelsOutcome/blood_transfusion_model.pkl')
# joblib.dump(list(X_train.columns), 'ModelsOutcome/blood_transfusion_feature_columns.pkl')
#
# print("model saved")
