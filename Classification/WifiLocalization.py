import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# خوندن فایل - چون با تب جدا شده و هدر نداره، باید این دو مورد رو مشخص کنیم
column_names = ['wifi_1', 'wifi_2', 'wifi_3', 'wifi_4', 'wifi_5', 'wifi_6', 'wifi_7', 'room']

df = pd.read_csv('Datasets/wifi_localization.txt', sep='\t', header=None, names=column_names)
# print(df.shape)
# print(df.head())
# print(df.dtypes)

# # بررسی آماری فقط روی ستون‌های وای‌فای (نه room، چون طبقه‌ایه)
# print(df.iloc[:, :-1].describe())
#
# # توزیع تعداد نمونه در هر اتاق
# print(df['room'].value_counts().sort_index())

# fig, axes = plt.subplots(3, 3, figsize=(15, 12))
# axes = axes.flatten()
#
# for i, col in enumerate(['wifi_1', 'wifi_2', 'wifi_3', 'wifi_4', 'wifi_5', 'wifi_6', 'wifi_7']):
#     sns.boxplot(data=df, x='room', y=col, ax=axes[i])
#     axes[i].set_title(col)
#
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.tight_layout()
# plt.show()

# def count_outliers_iqr(series):
#     Q1 = series.quantile(0.25)
#     Q3 = series.quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = series[(series < lower_bound) | (series > upper_bound)]
#     return len(outliers)
#
# for col in ['wifi_1', 'wifi_2', 'wifi_3', 'wifi_4', 'wifi_5', 'wifi_6', 'wifi_7']:
#     n_outliers = count_outliers_iqr(df[col])
#     print(f"{col}: {n_outliers} outlier ({n_outliers/len(df)*100:.1f}%)")

from sklearn.model_selection import train_test_split

X = df.drop('room', axis=1)
y = df['room']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)
# print(y_train.value_counts().sort_index())
# print(y_test.value_counts().sort_index())



wifi_cols = ['wifi_1', 'wifi_2', 'wifi_3', 'wifi_4', 'wifi_5', 'wifi_6', 'wifi_7']

# ایده ۱: تفاوت بین قوی‌ترین دو سیگنال
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()

sorted_train = np.sort(X_train_fe[wifi_cols].values, axis=1)
sorted_test = np.sort(X_test_fe[wifi_cols].values, axis=1)

X_train_fe['strongest_signal'] = sorted_train[:, -1]
X_train_fe['signal_range'] = sorted_train[:, -1] - sorted_train[:, 0]

X_test_fe['strongest_signal'] = sorted_test[:, -1]
X_test_fe['signal_range'] = sorted_test[:, -1] - sorted_test[:, 0]

# ایده ۲: کدوم روتر قوی‌ترینه (index)
X_train_fe['strongest_router'] = X_train[wifi_cols].values.argmax(axis=1)
X_test_fe['strongest_router'] = X_test[wifi_cols].values.argmax(axis=1)
# print(X_train_fe.head().to_string())
# print(X_train_fe.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# چون Logistic Regression و KNN به مقیاس حساسن، داده رو استاندارد می‌کنیم
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fe)
X_test_scaled = scaler.transform(X_test_fe)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    if name in ['Logistic Regression', 'KNN']:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train_fe, y_train)
        preds = model.predict(X_test_fe)

    acc = accuracy_score(y_test, preds)
    results[name] = acc
    # print(f"{name}: Accuracy = {acc:.4f}")

from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# for name, model in models.items():
#     if name in ['Logistic Regression', 'KNN']:
#         scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
#     else:
#         scores = cross_val_score(model, X_train_fe, y_train, cv=cv, scoring='accuracy')
#
#     print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
#     print(f"  تک‌تک فولدها: {np.round(scores, 4)}")

import joblib

# آموزش نهایی روی کل داده‌ی train (با فیچرهای مهندسی‌شده)
final_model = LogisticRegression(max_iter=1000, random_state=42)
final_model.fit(X_train_scaled, y_train)

# ارزیابی نهایی روی test (که تا الان دست‌نخورده بود)
final_preds = final_model.predict(X_test_scaled)
final_acc = accuracy_score(y_test, final_preds)
print(f"دقت نهایی روی Test Set: {final_acc:.4f}")

# ذخیره‌ی مدل و scaler با هم (چون هر دو برای پیش‌بینی جدید لازمن)
joblib.dump(final_model, 'ModelsOutcome/wifi_room_model.pkl')
joblib.dump(scaler, 'ModelsOutcome/wifi_room_scaler.pkl')

print("مدل و scaler ذخیره شدن.")