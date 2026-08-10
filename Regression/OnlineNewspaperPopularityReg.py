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

suspect_cols = ['n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens']
# mask = (df[suspect_cols] > 1).any(axis=1)
# print(mask.sum())
# print(df.loc[mask, suspect_cols + ['n_tokens_content']].to_string())
df = df.drop(index=31037).reset_index(drop=True)
# print(df.shape)
# print(df[suspect_cols].max())

X = df.drop(columns=['url', 'timedelta', 'shares'])
y = np.log1p(df['shares'])
# print(X.shape, y.shape)
# print(X.dtypes.value_counts())

# numeric_cols = X.select_dtypes(include=[np.number]).columns
# corr_with_target = X[numeric_cols].corrwith(y).sort_values(key=abs, ascending=False)
# print(corr_with_target.head(20))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape)
# print(X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
# print(X_train_scaled.describe().loc[['mean', 'std']].T.head())

# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

models = {
    # 'Ridge': (Ridge(random_state=42), X_train_scaled, X_test_scaled),
    # 'Random Forest': (RandomForestRegressor(random_state=42, n_jobs=-1), X_train, X_test),
    'Gradient Boosting': (GradientBoostingRegressor(random_state=42), X_train, X_test),
}
# results = []
# for name, (model, X_tr, X_te) in models.items():
#     cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')
#     model.fit(X_tr, y_train)
#     y_pred = model.predict(X_te)
#     from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#     results.append({
#         'Model': name,
#         'CV R2 (mean)': cv_scores.mean(),
#         'CV R2 (std)': cv_scores.std(),
#         'Test R2': r2_score(y_test, y_pred),
#         'Test RMSE': mean_squared_error(y_test, y_pred, squared=False),
#         'Test MAE': mean_absolute_error(y_test, y_pred)
#     })
# results_df = pd.DataFrame(results)
# print(results_df.to_string())



# medians = X_train.median()
# print(medians.to_dict())

# import joblib

final_model = GradientBoostingRegressor(random_state=42)
final_model.fit(X_train, y_train)
from sklearn.metrics import r2_score
y_pred_final = final_model.predict(X_test)
r2 = r2_score(y_test, y_pred_final)
print(f"R²: {r2:.4f}")

#
# joblib.dump(final_model, 'ModelsOutcome/online_news_popularity_gb_regressor.pkl')
#
# joblib.dump(list(X_train.columns), 'ModelsOutcome/online_news_popularity_reg_feature_columns.pkl')
#
# print("model saved")