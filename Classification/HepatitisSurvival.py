import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE',
           'MALAISE', 'ANOREXIA', 'LIVER_BIG', 'LIVER_FIRM', 'SPLEEN_PALPABLE',
           'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'ALK_PHOSPHATE',
           'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']

df = pd.read_csv('Datasets/hepatitis.data', header=None, names=columns, na_values='?')
# print(df.shape)
# print(df.dtypes)
# print(df.head())
# print(df.isnull().sum())
print(df.describe().to_string())
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Percent': missing_percent.round(2)
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
missing_summary.sort_values('Missing_Percent', ascending=False)
# print(missing_summary)
binary_cols = ['SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE',
               'ANOREXIA', 'LIVER_BIG', 'LIVER_FIRM', 'SPLEEN_PALPABLE',
               'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']
df_display = df.copy()

for col in binary_cols:
    if col == 'SEX':
        df_display[col] = df_display[col].map({1: 'male', 2: 'female'})
    else:
        df_display[col] = df_display[col].map({1: 'no', 2: 'yes'})

df_display['Class'] = df_display['Class'].map({1: 'DIE', 2: 'LIVE'})
# print(df_display.head())

class_counts = df_display['Class'].value_counts()
class_percent = df_display['Class'].value_counts(normalize=True) * 100
# print(class_counts)
# print(class_percent.round(2))
#
# plt.figure(figsize=(6,4))
# class_counts.plot(kind='bar', color=['steelblue', 'salmon'])
# plt.title('Distribution of Class (DIE vs LIVE)')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.xticks(rotation=0)
# plt.show()

numerical_cols = ['AGE', 'BILIRUBIN', 'ALK_PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME']
# print(df_display[numerical_cols].describe())
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()
# for i, col in enumerate(numerical_cols):
#     axes[i].hist(df_display[col].dropna(), bins=20, color='steelblue', edgecolor='black')
#     axes[i].set_title(col)
#     axes[i].set_xlabel(col)
#     axes[i].set_ylabel('Frequency')
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()
# for i, col in enumerate(numerical_cols):
#     df_display.boxplot(column=col, by='Class', ax=axes[i])
#     axes[i].set_title(col)
#     axes[i].set_xlabel('Class')
# plt.suptitle('')
# plt.tight_layout()
# plt.show()

# df_corr = df.copy()
# df_corr['Class_numeric'] = df['Class'].map({1: 0, 2: 1})
# corr_cols = numerical_cols + ['Class_numeric']
# correlation_matrix = df_corr[corr_cols].corr()
# print(correlation_matrix['Class_numeric'].sort_values(ascending=False))
# import seaborn as sns
# plt.figure(figsize=(9, 7))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
# plt.title('Correlation Matrix')
# plt.tight_layout()
# plt.show()

df_clean = df.copy()
# ساخت missing flag برای دو ستونی که بیشترین missing رو دارن
df_clean['PROTIME_was_missing'] = df_clean['PROTIME'].isnull().astype(int)
df_clean['ALK_PHOSPHATE_was_missing'] = df_clean['ALK_PHOSPHATE'].isnull().astype(int)
binary_cols_impute = ['STEROID', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER_BIG',
                       'LIVER_FIRM', 'SPLEEN_PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES']
#Impute by mode
for col in binary_cols_impute:
    mode_value = df_clean[col].mode()[0]
    df_clean[col] = df_clean[col].fillna(mode_value)
#impute by median
numerical_impute = ['BILIRUBIN', 'ALK_PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME']
for col in numerical_impute:
    median_value = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_value)
# print(df_clean.isnull().sum().sum())
# print(df_clean.info())

# پیدا کردن سطرهایی که PROTIME صفر یا خیلی نزدیک صفر دارن
suspicious = df_clean[df_clean['PROTIME'] < 5]
# print(suspicious[['Class', 'PROTIME', 'PROTIME_was_missing']])
median_protime = df_clean['PROTIME'].median()
df_clean.loc[df_clean['PROTIME'] < 5, 'PROTIME'] = median_protime
df_clean.loc[130, 'PROTIME_was_missing'] = 1  # imputed flag
# print(df_clean.loc[130, ['Class', 'PROTIME', 'PROTIME_was_missing']])

from sklearn.model_selection import train_test_split

X = df_clean.drop('Class', axis=1)
y = df_clean['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

svm_base = SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm_calibrated = CalibratedClassifierCV(svm_base, ensemble=False)
models = {
    'Logistic Regression': (LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000), X_train_scaled, X_test_scaled),
    'SVM': (svm_calibrated, X_train_scaled, X_test_scaled),
    'Random Forest': (RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), X_train, X_test)
}
predictions = {}
for name, (model, X_tr, X_te) in models.items():
    model.fit(X_tr, y_train)
    predictions[name] = model.predict(X_te)
    acc = accuracy_score(y_test, predictions[name])
    f1 = f1_score(y_test, predictions[name])
    print(f"{name}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

from sklearn.metrics import confusion_matrix, classification_report
# for name in models.keys():
#     print(f"===== {name} =====")
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, predictions[name]))
#     print()
#     print("Classification Report:")
#     print(classification_report(y_test, predictions[name], target_names=['DIE', 'LIVE']))
#     print()

# from sklearn.model_selection import StratifiedKFold, cross_val_score
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# for name, (model, X_full, _) in models.items():
#     X_for_cv = X_train_scaled if name in ['Logistic Regression', 'SVM'] else X_train
#     scores_f1 = cross_val_score(model, X_for_cv, y_train, cv=skf, scoring='f1_macro')
#     scores_recall = cross_val_score(model, X_for_cv, y_train, cv=skf, scoring='recall_macro')
#     print(f"{name}:")
#     print(f"  F1-macro: {scores_f1.mean():.4f} (+/- {scores_f1.std():.4f})")
#     print(f"  Recall-macro: {scores_recall.mean():.4f} (+/- {scores_recall.std():.4f})")

import joblib

final_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
final_model.fit(X, y)

# joblib.dump(final_model, 'ModelsOutcome/hepatitis_rf_model.pkl')
# feature_columns = X.columns.tolist()
# joblib.dump(feature_columns, 'ModelsOutcome/hepatitis_feature.pkl')
# print("model saved")
