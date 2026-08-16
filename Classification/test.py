import joblib
model = joblib.load("ModelsOutcome/online_news_popularity_gb_model.pkl")
print(model)
features = joblib.load("ModelsOutcome/online_news_popularity_feature_columns.pkl")
print(features)
#
# encoder = joblib.load("ModelsOutcome/autism_scaler.pkl")
# print(encoder)


import pandas as pd
import numpy as np

df = pd.read_csv('Datasets/OnlineNewsPopularity.csv')

df.columns = df.columns.str.strip()

df['popular'] = (df['shares'] >= 1400).astype(int)

suspect_cols = ['n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens']

df = df.drop(index=31037).reset_index(drop=True)

X = df.drop(columns=['url', 'timedelta', 'shares', 'popular'])
y = df['popular']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

models = {
    # 'Logistic Regression': (LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42), X_train_scaled, X_test_scaled),
    # 'Random Forest': (RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1), X_train, X_test),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), X_train, X_test),
}
# results = []
# for name, (model, X_tr, X_te) in models.items():
#     cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
#     model.fit(X_tr, y_train)
#     y_pred = model.predict(X_te)
#     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#     results.append({
#         'Model': name,
#         'CV Accuracy (mean)': cv_scores.mean(),
#         'CV Accuracy (std)': cv_scores.std(),
#         'Test Accuracy': accuracy_score(y_test, y_pred),
#         'Test Precision': precision_score(y_test, y_pred),
#         'Test Recall': recall_score(y_test, y_pred),
#         'Test F1': f1_score(y_test, y_pred)
#     })
# results_df = pd.DataFrame(results)
# print(results_df.to_string())

import joblib

final_model = GradientBoostingClassifier(random_state=42)
final_model.fit(X_train, y_train)

joblib.dump(final_model, 'ModelsOutcome/online_news_popularity_gb_model.pkl')

joblib.dump(list(X_train.columns), 'ModelsOutcome/online_news_popularity_feature_columns.pkl')

print("model saved")


