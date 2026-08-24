import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


columns = ["Season", "Age", "Childish_Diseases", "Accident_Trauma","Surgical_Intervention", "High_Fevers_LastYear",
    "Alcohol_Consumption", "Smoking_Habit","Hours_Sitting_PerDay", "Diagnosis"]

df = pd.read_csv('Datasets/fertility_Diagnosis.txt',header=None, names=columns)
features = ['Season', 'Age', 'Childish_Diseases', 'Accident_Trauma', 'Surgical_Intervention',
            "High_Fevers_LastYear",'Alcohol_Consumption', 'Smoking_Habit','Hours_Sitting_PerDay']
# print(df.head().to_string())
# print(df.describe())
# print(df.shape)
# print(df.isnull().sum())
#
# fig,ax= plt.subplots(3,3,figsize=(10,15))
# axes =ax.flatten()
# for i,col in enumerate(features):
#     axes[i].hist(df[col], bins=10, color='blue',edgecolor='black')
#     axes[i].set_xlabel(col)
#
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,15))
# df[features].boxplot()
# plt.show()

# plt.figure(figsize = (10,15))
# corr_matrix = df[features].corr()
# sns.heatmap(corr_matrix, annot=True, cmap="Blues", vmin=0, vmax=1)
# plt.show()

# df['Diagnosis_num'] = df['Diagnosis'].map({'N': 0, 'O': 1})
# diagnosis =df.groupby('Diagnosis')[['Season','Age','Childish_Diseases','Accident_Trauma',
#                           'Surgical_Intervention','High_Fevers_LastYear',
#                           'Alcohol_Consumption','Smoking_Habit',
#                           'Hours_Sitting_PerDay']].mean()
# print(diagnosis.to_string())

from sklearn.model_selection import train_test_split
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis'].map({'N': 0, 'O': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

models = {
    'Logistic Regression': (LogisticRegression(random_state=42, class_weight='balanced'), X_train, X_test),
    'SVM': (SVC(random_state=42, class_weight='balanced'), X_train, X_test),
    'Random Forest': (RandomForestClassifier(random_state=42, class_weight='balanced'), X_train, X_test),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), X_train, X_test),  # GB از class_weight پشتیبانی نمی‌کنه
}
# results = []
# for name, (model, X_train, X_test) in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     results.append({
#         'Model': name,
#         'Accuracy': accuracy_score(y_test, y_pred),
#         'Precision': precision_score(y_test, y_pred),
#         'Recall': recall_score(y_test, y_pred),
#         'F1': f1_score(y_test, y_pred)
#     })
# results_df = pd.DataFrame(results)
# print(results_df)

# conf_matrix =confusion_matrix(y_test, y_pred)

from sklearn.model_selection import  cross_val_score, StratifiedKFold

cv= StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
# cv_results =[]
# for name, (model, X_train, X_test) in models.items():
#     scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
#     cv_results.append({
#         'Model': name,
#         'cv f1 mean' :scores.mean(),
#         'cv f1 std' :scores.std(),
#         'stability mean- std' :scores.mean()-scores.std(),
#     })
#
# cv_df = pd.DataFrame(cv_results)
# print(cv_df)  #svm is the best

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

for name, (model, X_train, X_test) in models.items():
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)  # توجه: روی کل X, y نه فقط X_train
#     print(f"\n--- {name} ---")
#     print(confusion_matrix(y, y_pred_cv))
#     print(classification_report(y, y_pred_cv, target_names=['N', 'O']))

# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'C':[0.0001, 0.001, 0.01, 0.1, 1, 10]
# }
#
# grid_search = GridSearchCV(
#     estimator=LogisticRegression(random_state=42, class_weight='balanced'),
#     param_grid=param_grid,
#     cv=cv,               # همون StratifiedKFold قبلی
#     scoring='recall'      # چون طبق بحث قبلی، Recall برای این مسئله اولویت داره
# )
#
# grid_search.fit(X, y)   # روی کل دیتا (نه فقط X_train) چون خودش داخلی CV می‌کنه
#
# print("بهترین C:", grid_search.best_params_)
# print("بهترین Recall:", grid_search.best_score_)
# best_model = grid_search.best_estimator_
# y_pred_cv = cross_val_predict(best_model, X, y, cv=cv)
# print(classification_report(y, y_pred_cv, target_names=['N', 'O']))

final_model = LogisticRegression(random_state=42, class_weight='balanced')
final_model.fit(X,y)

# print("عملکرد نهایی (بر اساس Cross-Validation):")
# print(classification_report(y, cross_val_predict(final_model, X, y, cv=cv), target_names=['N', 'O']))

import joblib

joblib.dump(final_model, 'ModelsOutcome/fertility_lr_model.pkl')
joblib.dump(list(X_train.columns), 'ModelsOutcome/fertility_feature_columns.pkl')