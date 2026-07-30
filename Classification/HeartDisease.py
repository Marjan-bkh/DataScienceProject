import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "Datasets/processed.cleveland.data"

# نام ستون‌ها رو دستی تعریف می‌کنیم چون فایل header نداره
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# na_values='?' یعنی: هر جا علامت ? دیدی، به NaN تبدیلش کن
df = pd.read_csv(file_path, header=None, names=columns, na_values='?')
# print(df.head().to_string())
# print(df.info())
# print(df.describe())

# حالا target رو به باینری تبدیل می‌کنیم: 0 = سالم، هر چیز بزرگتر از 0 = بیمار
df['target'] = (df['target'] > 0).astype(int)
# بررسی توزیع کلاس‌ها بعد از تبدیل
# print(df['target'].value_counts())

# # تنظیم استایل کلی نمودارها
# sns.set_style("whitegrid")
#
# # 1. هیستوگرام متغیرهای عددی پیوسته
# numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
# df[numeric_cols].hist(figsize=(12, 8), bins=20, edgecolor='black')
# plt.suptitle("توزیع متغیرهای عددی")
# plt.tight_layout()
# plt.show()
#
# # 2. نمودار میله‌ای برای هر متغیر categorical در برابر target
# categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
# fig, axes = plt.subplots(4, 2, figsize=(14, 16))
# axes = axes.flatten()
# for i, col in enumerate(categorical_cols):
#     sns.countplot(data=df, x=col, hue='target', ax=axes[i])
#     axes[i].set_title(f'{col} بر اساس target')
# plt.tight_layout()
# plt.show()
#
# # 3. مقایسه‌ی سن و حداکثر ضربان قلب بین دو گروه با boxplot
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# sns.boxplot(data=df, x='target', y='age', ax=axes[0])
# axes[0].set_title('سن بر اساس target')
# sns.boxplot(data=df, x='target', y='thalach', ax=axes[1])
# axes[1].set_title('حداکثر ضربان قلب بر اساس target')
# plt.tight_layout()
# plt.show()
#
# # 4. ماتریس همبستگی
# plt.figure(figsize=(12, 10))
# corr = df.corr()
# sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
# plt.title("ماتریس همبستگی")
# plt.tight_layout()
# plt.show()
#
# # همبستگی هر ویژگی با target به‌صورت مرتب‌شده (برای دید بهتر)
# print(corr['target'].sort_values(ascending=False))

# نمایش کامل ردیف‌هایی که missing value دارن
missing_rows = df[df['ca'].isnull() | df['thal'].isnull()]
# print(missing_rows)
df_clean = df.dropna(subset=['ca', 'thal']).reset_index(drop=True)

# def detect_outliers_iqr(data, column):
#     Q1 = data[column].quantile(0.25)
#     Q3 = data[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
#     return outliers, lower_bound, upper_bound
#
# # فقط روی متغیرهای عددی پیوسته چک می‌کنیم (نه categorical)
# continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
#
# for col in continuous_cols:
#     outliers, lb, ub = detect_outliers_iqr(df_clean, col)
#     print(f"{col}: تعداد outlier = {len(outliers)}  |  محدوده مجاز = [{lb:.2f}, {ub:.2f}]")

from sklearn.model_selection import train_test_split

# جدا کردن ویژگی‌ها (X) از متغیر هدف (y)
X = df_clean.drop('target', axis=1)
y = df_clean['target']

# تقسیم داده: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% داده برای تست کنار گذاشته می‌شه
    random_state=42,     # عدد ثابت برای اینکه هر بار نتیجه‌ی یکسان تکرار بشه
    stratify=y           # نسبت کلاس‌ها رو در train/test حفظ می‌کنه
)

# ستون‌هایی که باید one-hot بشن
categorical_features = ['cp', 'restecg', 'slope', 'thal']

# one-hot encoding با pandas
X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# چون ممکنه بعضی دسته‌ها فقط توی train یا فقط توی test باشن،
# باید ستون‌های X_test رو دقیقاً با X_train یکی کنیم
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# print("ستون‌های قبل از encoding:", X_train.shape[1])
# print("ستون‌های بعد از encoding:", X_train_encoded.shape[1])
# print()
# print(X_train_encoded.columns.tolist())
# print()
# print(X_train_encoded.head())
from sklearn.preprocessing import StandardScaler

# فقط روی ستون‌های عددی پیوسته (نه دامی‌های 0/1) اسکیل می‌کنیم
# چون دامی‌ها از قبل بین 0 و 1 هستن و نیازی به تغییر ندارن
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

scaler = StandardScaler()

# نسخه‌ی کپی می‌سازیم تا نسخه‌ی اصلی encoded دست‌نخورده بمونه
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()

# مهم: scaler رو فقط با train "fit" می‌کنیم (یاد می‌گیره mean و std رو از train)
X_train_scaled[numeric_features] = scaler.fit_transform(X_train_encoded[numeric_features])

# روی test فقط "transform" می‌کنیم (با همون mean/std که از train یاد گرفته، نه دوباره fit)
X_test_scaled[numeric_features] = scaler.transform(X_test_encoded[numeric_features])

# print(X_train_scaled[numeric_features].describe())

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
}

results = []

for name, model in models.items():
    if name == 'Logistic Regression':
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train_encoded, X_test_encoded

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]  # احتمال کلاس 1 (بیمار)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    })
#
# results_df = pd.DataFrame(results)
# print(results_df.to_string(index=False))

from sklearn.model_selection import cross_val_score, StratifiedKFold

# StratifiedKFold مثل stratify توی train_test_split عمل می‌کنه:
# نسبت کلاس‌ها (سالم/بیمار) رو توی هر فولد حفظ می‌کنه
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# cv_results = []
#
# for name, model in models.items():
#     if name == 'Logistic Regression':
#         X_full = pd.concat([X_train_scaled, X_test_scaled])
#     else:
#         X_full = pd.concat([X_train_encoded, X_test_encoded])
#
#     y_full = pd.concat([y_train, y_test])
#     acc_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='accuracy')
#     auc_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='roc_auc')
#
#     cv_results.append({
#         'Model': name,
#         'Accuracy Mean': acc_scores.mean(),
#         'Accuracy Std': acc_scores.std(),
#         'ROC-AUC Mean': auc_scores.mean(),
#         'ROC-AUC Std': auc_scores.std()
#     })
#
# cv_results_df = pd.DataFrame(cv_results)
# print(cv_results_df.to_string(index=False))

# مدل Logistic Regression که قبلاً train شده رو از دیکشنری models می‌گیریم
log_reg_model = models['Logistic Regression']

# استخراج ضرایب (coefficients) به همراه اسم فیچرها
coefficients = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Coefficient': log_reg_model.coef_[0]
})

# مرتب‌سازی بر اساس قدر مطلق (بزرگ‌ترین تأثیر چه مثبت چه منفی) بالا باشه
coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

# print(coefficients[['Feature', 'Coefficient']].to_string(index=False))
#
# # نمودار میله‌ای برای دید بهتر
# plt.figure(figsize=(10, 8))
# colors = ['red' if c > 0 else 'blue' for c in coefficients['Coefficient']]
# plt.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors)
# plt.xlabel('مقدار ضریب (Coefficient)')
# plt.title('اهمیت فیچرها در پیش‌بینی بیماری قلبی (Logistic Regression)')
# plt.axvline(x=0, color='black', linewidth=0.8)
# plt.gca().invert_yaxis()  # بزرگ‌ترین بالا باشه
# plt.tight_layout()
# plt.show()

import joblib

# ذخیره‌ی مدل، اسکیلر، و اسم ستون‌ها در یک فایل واحد (dictionary)
model_package = {
    'model': log_reg_model,
    'scaler': scaler,
    'feature_columns': X_train_scaled.columns.tolist(),
    'numeric_features': numeric_features,      # ستون‌هایی که باید scale بشن
    'categorical_features': categorical_features  # ستون‌هایی که باید one-hot بشن
}

joblib.dump(model_package, 'ModelsOutcome/heart_disease_model.pkl')

print("مدل با موفقیت ذخیره شد در: heart_disease_model.pkl")

# لود کردن مدل ذخیره‌شده
loaded_package = joblib.load('ModelsOutcome/heart_disease_model.pkl')

loaded_model = loaded_package['model']
loaded_scaler = loaded_package['scaler']

# مثال: پیش‌بینی برای یک بیمار جدید (بعد از انجام همون preprocessing)
# prediction = loaded_model.predict(new_patient_data_scaled)