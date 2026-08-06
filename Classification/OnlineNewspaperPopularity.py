import pandas as pd
import numpy as np

df = pd.read_csv('Datasets/OnlineNewsPopularity.csv')
# print(df.shape)
# print(df.columns.tolist())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum().sum())

df.columns = df.columns.str.strip()
# print(df.columns.tolist())

#  (duplicate articles)
# print(df['url'].duplicated().sum())
# print("Duplicate rows (excluding url):", df.drop(columns='url').duplicated().sum())

df['popular'] = (df['shares'] >= 1400).astype(int)
# print(df['popular'].value_counts())
# print(df['popular'].value_counts(normalize=True))
# print(df['shares'].sort_values(ascending=False).head(10))

import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# axes[0].hist(df['shares'], bins=100)
# axes[0].set_title('Distribution of shares (raw)')
# axes[0].set_xlabel('shares')
# axes[1].hist(np.log1p(df['shares']), bins=100)
# axes[1].set_title('Distribution of log(shares+1)')
# axes[1].set_xlabel('log(shares+1)')
# plt.tight_layout()
# plt.show()

# numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['shares', 'popular'])
# corr_with_target = df[numeric_cols].corrwith(df['popular']).sort_values(key=abs, ascending=False)
# print(corr_with_target.head(20))

suspect_cols = ['n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens']
# mask = (df[suspect_cols] > 1).any(axis=1)
# print(mask.sum())
# print(df.loc[mask, suspect_cols + ['n_tokens_content']].to_string())
df = df.drop(index=31037).reset_index(drop=True)
# print(df.shape)
# print(df[suspect_cols].max())

X = df.drop(columns=['url', 'timedelta', 'shares', 'popular'])
y = df['popular']
# print(X.shape, y.shape)
# print(X.dtypes.value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.value_counts(normalize=True))
# print(y_test.value_counts(normalize=True))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
# print(X_train_scaled.describe().loc[['mean', 'std']].T.head())

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

