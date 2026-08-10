import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

column_names = ['IR', 'MR', 'FF', 'CR', 'CO', 'OP', 'Class']
#Industrial Risk (IR),Management Risk (MR),Financial Flexibility (FF),Credibility (CR),Competitiveness (CO),Operating Risk (OP)
df = pd.read_csv('Datasets/Qualitative_Bankruptcy.data.txt', header=None, names=column_names)
# print(df.shape)
# print(df.head(10))
# print(df.info())
# print(df['Class'].value_counts())

features = ['IR', 'MR', 'FF', 'CR', 'CO', 'OP']

# for col in features:
#     print(f"--- {col} ---")
#     print(df[col].value_counts())
#     print()

# for col in features:
#     print(f"--- {col} vs Class ---")
#     print(pd.crosstab(df[col], df['Class']))
#     print()

# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()
#
# for i, col in enumerate(features):
#     pd.crosstab(df[col], df['Class']).plot(kind='bar', ax=axes[i])
#     axes[i].set_title(f'{col} vs Class')
#     axes[i].set_xlabel(col)
#     axes[i].set_ylabel('Count')
#     axes[i].legend(title='Class')
# plt.tight_layout()
# plt.show()

# # N بدترین=0, A متوسط=1, P بهترین=2
risk_mapping = {'N': 0, 'A': 1, 'P': 2}
df_encoded = df.copy()
for col in features:
    df_encoded[col] = df_encoded[col].map(risk_mapping)

df_encoded['Class'] = df_encoded['Class'].map({'NB': 0, 'B': 1})
# print(df_encoded.head(10))
# print(df_encoded.dtypes)

# print(df_encoded.corr()['Class'].sort_values(ascending=False))

from sklearn.model_selection import train_test_split

X = df_encoded[features]   # ورودی‌ها: IR, MR, FF, CR, CO, OP
y = df_encoded['Class']    # هدف: 0 یا 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print(y_train.value_counts(normalize=True))
# print(y_test.value_counts(normalize=True))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# results = {}
#
# for name, model in models.items():
#
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#
#     # محاسبه معیارها
#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred)
#     rec = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#
#     results[name] = {
#         'Accuracy': acc,
#         'Precision': prec,
#         'Recall': rec,
#         'F1-Score': f1
#     }
#
#     print(f"=== {name} ===")
#     print(f"Accuracy:  {acc:.4f}")
#     print(f"Precision: {prec:.4f}")
#     print(f"Recall:    {rec:.4f}")
#     print(f"F1-Score:  {f1:.4f}")
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))
#     print()

# results_df = pd.DataFrame(results).T
# print(results_df)

from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# for name, model in models.items():
#     scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
#     print(f"{name}:")
#     print(f"  دقت هر fold: {scores}")
#     print(f"  میانگین: {scores.mean():.4f}  |  انحراف معیار: {scores.std():.4f}")
#     print()

import joblib

rf_production = RandomForestClassifier(random_state=42)
rf_production.fit(X, y)

#
# joblib.dump(rf_production, 'ModelsOutcome/bankruptcy_model.pkl')
#
# joblib.dump(risk_mapping, 'ModelsOutcome/bankruptcy_risk_mapping.pkl')

# joblib.dump(list(X_train.columns), 'ModelsOutcome/bankruptcy_feature.pkl')
# print("model saved")
#
# loaded_model = joblib.load('ModelsOutcome/bankruptcy_model.pkl')
#
# #sample test
# sample = pd.DataFrame([[1, 1, 0, 0, 0, 1]], columns=features)  # IR=A, MR=A, FF=N, CR=N, CO=N, OP=A
# prediction = loaded_model.predict(sample)
# prediction_proba = loaded_model.predict_proba(sample)
#
# print("پیش‌بینی:", "Bankruptcy" if prediction[0] == 1 else "Non-Bankruptcy")
# print("احتمال هر کلاس [NB, B]:", prediction_proba)