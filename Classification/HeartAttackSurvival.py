import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

column_names = [
    'survival',
    'still_alive',
    'age_at_heart_attack',
    'pericardial_effusion',
    'fractional_shortening',
    'epss',
    'lvdd',
    'wall_motion_score',
    'wall_motion_index',
    'mult',
    'name',
    'group',
    'alive_at_1'
]
df = pd.read_csv(
    'Datasets/echocardiogram.data',
    header=None,
    names=column_names,
    na_values='?',
    on_bad_lines='skip',   # سطرهایی که تعداد فیلدشون درست نیست رو رد کن
    engine='python'        # پارامتر on_bad_lines='skip' با این engine بهتر کار می‌کنه
)
# print(df.shape)
# print(df.head(10))
# print(df.dtypes)

# --- روش اولیه‌ی ساخت target (survival >= 12 ماه) - کنار گذاشته شد چون کلاس ۰ فقط ۴ نمونه داشت ---
# df_clean = df.drop(columns=['name', 'mult', 'group', 'wall_motion_score'])
# # print(df_clean.shape)
# # print(df_clean.columns.tolist())
#
# def make_target(row):
#     if pd.isna(row['survival']) or pd.isna(row['still_alive']):
#         return np.nan
#     if row['survival'] >= 12:
#         return 1
#     elif row['still_alive'] == 0:
#         return 0
#     else:
#         return np.nan
#
# df_clean['target'] = df_clean.apply(make_target, axis=1)
# print(df_clean['target'].value_counts(dropna=False))

# تصمیم نهایی: target = still_alive (توزیع کلاس‌ها متعادل‌تر و منطقی‌تره)
df_final = df.drop(columns=['name', 'mult', 'group', 'wall_motion_score'])
df_final = df_final.drop(columns=['survival', 'alive_at_1'])
df_final = df_final.rename(columns={'still_alive': 'target'})
df_final = df_final.dropna(subset=['target'])
# print(df_final.shape)
# print(df_final.columns.tolist())
# print(df_final['target'].value_counts())
# print(df_final.isna().sum())

# features فقط برای نمودارهای EDA زیر لازم بود
# features = df_final.drop('target', axis=1)

# --- EDA: هیستوگرام هر فیچر به تفکیک target ---
# fig, axes = plt.subplots(2,3, figsize=(14,10))
# axes = axes.flatten()
# for i,col in enumerate(features):
#     sns.histplot(data=df_final, x=col, hue='target', ax=axes[i],
#                  kde=True, stat='density', common_norm=False, multiple='layer', alpha=0.4)
#     axes[i].set_title(f'{col} by Target')
# plt.tight_layout()
# plt.show()

# --- EDA: Correlation Matrix ---
# plt.figure(figsize=(10,10))
# corr = df_final.corr()
# sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt='.2f', center=0)
# plt.tight_layout()
# plt.show()

# --- EDA: Boxplot هر فیچر به تفکیک target ---
# fig, ax = plt.subplots(2,3)
# ax = ax.flatten()
# for i,col in enumerate(features):
#     sns.boxplot(x='target', y=col, data=df_final, ax=ax[i])
# plt.tight_layout()
# plt.show()

# --- بررسی نسبت missing هر ستون به تفکیک target (فقط برای تصمیم‌گیری درباره‌ی imputation) ---
# cols_with_missing = ['age_at_heart_attack', 'fractional_shortening', 'epss', 'lvdd', 'wall_motion_index']
# for col in cols_with_missing:
#     missing_by_target = df_final.groupby('target')[col].apply(lambda x: x.isna().mean())
#     print(f"\n{col}:")
#     print(missing_by_target)
# print(df_final.groupby('target')['age_at_heart_attack'].mean())

from sklearn.impute import SimpleImputer
# قدم ۱: ستون نشانگر برای فیچرهایی که رابطه‌ی قوی‌تری با target داشتن
cols_to_flag = ['fractional_shortening', 'epss', 'lvdd']
for col in cols_to_flag:
    df_final[f'{col}_was_missing'] = df_final[col].isna().astype(int)

# قدم ۲: پر کردن ۵ ستون با میانه
cols_with_missing = ['age_at_heart_attack', 'fractional_shortening', 'epss', 'lvdd', 'wall_motion_index']
imputer = SimpleImputer(strategy='median')
df_final[cols_with_missing] = imputer.fit_transform(df_final[cols_with_missing])
# print(df_final.isna().sum())
# print(df_final.shape)
# print(df_final[[c for c in df_final.columns if 'was_missing' in c]].sum())

from sklearn.model_selection import train_test_split
X = df_final.drop(columns=['target'])
y = df_final['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# print("Train shape:", X_train.shape)
# print("Test shape:", X_test.shape)
# print(y_train.value_counts(normalize=True))
# print(y_test.value_counts(normalize=True))

# --- Feature Engineering + Scaling: فقط برای مقایسه‌ی Logistic Regression لازم بود ---
# مدل نهایی (Random Forest) روی X_train خام کار می‌کنه، پس این بخش الان استفاده‌ای نداره
# X_train_fe = X_train.copy()
# X_test_fe = X_test.copy()
# X_train_fe['epss_lvdd_ratio'] = X_train_fe['epss'] / X_train_fe['lvdd']
# X_test_fe['epss_lvdd_ratio'] = X_test_fe['epss'] / X_test_fe['lvdd']
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# numeric_cols = ['age_at_heart_attack', 'fractional_shortening', 'epss', 'lvdd', 'wall_motion_index', 'epss_lvdd_ratio']
# X_train_scaled = X_train_fe.copy()
# X_test_scaled = X_test_fe.copy()
# X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_fe[numeric_cols])
# X_test_scaled[numeric_cols] = scaler.transform(X_test_fe[numeric_cols])

# --- مقایسه‌ی اولیه‌ی ۳ مدل روی یه تک Test (نتیجه‌ش گمراه‌کننده بود، با CV جایگزین شد) ---
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#
# models = {
#     'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
#     'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
#     'Gradient Boosting': GradientBoostingClassifier(random_state=42, max_depth=3, n_estimators=100)
# }
# results = {}
# for name, model in models.items():
#     if name == 'Logistic Regression':
#         model.fit(X_train_scaled, y_train)
#         y_pred = model.predict(X_test_scaled)
#     else:
#         model.fit(X_train_fe, y_train)
#         y_pred = model.predict(X_test_fe)
#     results[name] = {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred),
#         'recall': recall_score(y_test, y_pred),
#         'f1': f1_score(y_test, y_pred)
#     }
# results_df = pd.DataFrame(results).T
# print(results_df)

# --- مقایسه‌ی همون ۳ مدل با Cross-Validation ---
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_results = []
# for name, model in models.items():
#     X_data = X_train_scaled if name == 'Logistic Regression' else X_train
#     scores = cross_val_score(model, X_data, y_train, cv=cv, scoring='f1')
#     cv_results.append({
#         'Model': name,
#         'Mean F1': scores.mean(),
#         'Std F1': scores.std(),
#         'all_scores': scores
#     })
# results_df = pd.DataFrame(cv_results)
# print(results_df.to_string(index=False))

# تصمیم نهایی از این مقایسه‌ها: Random Forest بهترین تعادل بین دقت و پایداری رو داشت
from sklearn.ensemble import RandomForestClassifier

final_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
final_model.fit(X_train, y_train)
y_pred_final = final_model.predict(X_test)
y_proba_final = final_model.predict_proba(X_test)[:, 1]

# --- ارزیابی تک‌باره روی Test (فقط مرجع؛ تصمیم اصلی بر پایه‌ی Cross-Validation پایین‌تر گرفته شد) ---
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# print("--- Random Forest - نتیجه‌ی نهایی روی Test ---")
# print(f"Accuracy:  {accuracy_score(y_test, y_pred_final):.3f}")
# print(f"Precision: {precision_score(y_test, y_pred_final):.3f}")
# print(f"Recall:    {recall_score(y_test, y_pred_final):.3f}")
# print(f"F1-score:  {f1_score(y_test, y_pred_final):.3f}")
# importances = final_model.feature_importances_
# feature_names = X_train.columns
# importances_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances}).sort_values(by='Importance', ascending=False)
# print(importances_df.to_string(index=False))

# --- مقایسه‌ی نهایی و قابل‌اعتماد: با class_weight در مقابل بدون آن، با Cross-Validation ---
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_no_weight = RandomForestClassifier(random_state=42, n_estimators=100)
model_balanced = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')

scores_no_weight = cross_val_score(model_no_weight, X_train, y_train, cv=cv, scoring='f1')
scores_balanced = cross_val_score(model_balanced, X_train, y_train, cv=cv, scoring='f1')

# print("بدون class_weight:")
# print(f"  Mean F1: {scores_no_weight.mean():.3f}, Std: {scores_no_weight.std():.3f}")
#
# print("\nبا class_weight='balanced':")
# print(f"  Mean F1: {scores_balanced.mean():.3f}, Std: {scores_balanced.std():.3f}")

scores_no_weight_recall = cross_val_score(model_no_weight, X_train, y_train, cv=cv, scoring='recall')
scores_balanced_recall = cross_val_score(model_balanced, X_train, y_train, cv=cv, scoring='recall')

#print(f"\nRecall بدون weight: {scores_no_weight_recall.mean():.3f}")
#print(f"Recall با weight: {scores_balanced_recall.mean():.3f}")

import joblib
joblib.dump(final_model, 'ModelsOutcome/echocardiogram_rf_model.pkl')
feature_columns = X_train.columns.tolist()
joblib.dump(feature_columns, 'ModelsOutcome/echocardiogram_feature_columns.pkl')
print("مدل با موفقیت ذخیره شد.")
#print(f"تعداد فیچرها: {len(feature_columns)}")
#print(f"فیچرها: {feature_columns}")