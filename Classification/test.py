import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['wifi_1', 'wifi_2', 'wifi_3', 'wifi_4', 'wifi_5', 'wifi_6', 'wifi_7', 'room']

df = pd.read_csv('Datasets/wifi_localization.txt', sep='\t', header=None, names=column_names)

from sklearn.model_selection import train_test_split

X = df.drop('room', axis=1)
y = df['room']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)




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

final_model = LogisticRegression(max_iter=1000, random_state=42)
final_model.fit(X_train_scaled, y_train)

final_preds = final_model.predict(X_test_scaled)
final_acc = accuracy_score(y_test, final_preds)
print(f"دقت نهایی روی Test Set: {final_acc:.4f}")

joblib.dump(final_model, 'ModelsOutcome/wifi_room_model.pkl')
joblib.dump(scaler, 'ModelsOutcome/wifi_room_scaler.pkl')

print("model saved")