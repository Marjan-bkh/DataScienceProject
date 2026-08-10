import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# چون فایل car.data هدر (اسم ستون) نداره، خودمون اسم ستون‌ها رو مشخص می‌کنیم
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

df = pd.read_csv('Datasets/car.data', names=column_names)
# print(df.shape)
# print(df.head())
# print(df.info())
# # بررسی مقادیر یکتا در هر ستون
# for col in df.columns:
#     print(f"--- {col} ---")
#     print(df[col].value_counts())
#     print()
# plt.figure(figsize=(8, 5))
# sns.countplot(data=df, x='class', order=df['class'].value_counts().index, palette='viridis')
# plt.title('توزیع کلاس هدف (Class Distribution)')
# plt.xlabel('کلاس')
# plt.ylabel('تعداد نمونه')
#
# # نمایش درصد روی هر ستون
# total = len(df)
# for p in plt.gca().patches:
#     percentage = f'{100 * p.get_height() / total:.1f}%'
#     plt.gca().annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
#                         ha='center', va='bottom')
# plt.tight_layout()
# plt.show()

features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
#
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# axes = axes.flatten()
#
# for i, col in enumerate(features):
#     cross_tab = pd.crosstab(df[col], df['class'])
#     cross_tab.plot(kind='bar', stacked=True, ax=axes[i], colormap='viridis')
#     axes[i].set_title(f'{col} بر اساس class')
#     axes[i].set_xlabel(col)
#     axes[i].set_ylabel('تعداد')
#     axes[i].legend(title='class', fontsize=8)
#     axes[i].tick_params(axis='x', rotation=0)
#
# plt.tight_layout()
# plt.show()

from sklearn.model_selection import train_test_split
X = df.drop('class', axis=1)
Y = df['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0 , stratify=Y)
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.value_counts(normalize=True) * 100)
# print(Y_test.value_counts(normalize=True) * 100)
from sklearn.preprocessing import OrdinalEncoder

category_orders = [
    ['low', 'med', 'high', 'vhigh'],      # buying
    ['low', 'med', 'high', 'vhigh'],      # maint
    ['2', '3', '4', '5more'],             # doors
    ['2', '4', 'more'],                   # persons
    ['small', 'med', 'big'],              # lug_boot
    ['low', 'med', 'high']                # safety
]

encoder = OrdinalEncoder(categories=category_orders)

X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)
X_train_encoded = pd.DataFrame(X_train_encoded, columns=X_train.columns, index=X_train.index)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=X_test.columns, index=X_test.index)
# print(X_train_encoded.head())

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}
#
# results = {}
#
# for name, model in models.items():
#     model.fit(X_train_encoded, Y_train)

#     Y_pred = model.predict(X_test_encoded)

#     acc = accuracy_score(Y_test, Y_pred)
#     f1_macro = f1_score(Y_test, Y_pred, average='macro')
#
#     results[name] = {'accuracy': acc, 'f1_macro': f1_macro}
#
#     print(f"===== {name} =====")
#     print(f"Accuracy: {acc:.4f}")
#     print(f"F1-macro: {f1_macro:.4f}")
#     print(classification_report(Y_test, Y_pred))
#     print()

from sklearn.model_selection import StratifiedKFold, cross_val_score
#

# # (دقیقاً مثل stratify=y در train_test_split، ولی برای Cross-Validation)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
# cv_results = {}
#
# for name, model in models.items():
#     scores = cross_val_score(
#         model,
#         X_train_encoded,
#         Y_train,
#         cv=skf,
#         scoring='f1_macro'
#     )
#     cv_results[name] = scores
#     print(f"===== {name} =====")
#     print(f"امتیاز هر Fold: {np.round(scores, 4)}")
#     print(f"میانگین F1-macro: {scores.mean():.4f}")
#     print(f"انحراف معیار: {scores.std():.4f}")
#     print()

import joblib

final_model = GradientBoostingClassifier(random_state=42)
final_model.fit(X_train_encoded, Y_train)


y_test_pred = final_model.predict(X_test_encoded)
print(classification_report(Y_test, y_test_pred))


joblib.dump(final_model, 'ModelsOutcome/car_evaluation_gb_model.pkl')
joblib.dump(encoder, 'ModelsOutcome/car_evaluation_encoder.pkl')
joblib.dump(list(X_train_encoded.columns), 'ModelsOutcome/car_evaluation_feature.pkl')

print("model saved")