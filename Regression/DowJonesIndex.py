import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('DataSets/dow_jones_index.data')
# print(df.head().to_string())
# print(df.describe().to_string())
# print(df.shape)
# print(df.isnull().sum())
# print(df.dtypes)

price_cols = ['open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close']
for col in price_cols:
    df[col] = df[col].str.replace('$', '', regex=False).astype(float)
df['date'] = pd.to_datetime(df['date'])
# print(df.dtypes)
# print(df[['open', 'high', 'low', 'close', 'date']].head())

# print(df[df['previous_weeks_volume'].isnull()][['quarter', 'stock', 'date']])
df = df[df['previous_weeks_volume'].notnull()].reset_index(drop=True)
# print(df.isnull().sum())
# print(df.shape)

# print(df['percent_change_next_weeks_price'].describe())
#
# plt.figure(figsize=(10, 5))
# sns.histplot(df['percent_change_next_weeks_price'], bins=30, kde=True)
# plt.title('Distribution of Percent Change Next Week Price')
# plt.xlabel('Percent Change (%)')
# plt.show()
#
# plt.figure(figsize=(10, 5))
# sns.boxplot(x=df['percent_change_next_weeks_price'])
# plt.title('Boxplot of Percent Change Next Week Price')
# plt.show()

# numeric_cols = df.select_dtypes(include=[np.number]).columns
# corr_matrix = df[numeric_cols].corr()
# corr_matrix['percent_change_next_weeks_price'].sort_values(ascending=False)
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.tight_layout()
# plt.show()

df = df.sort_values(['stock', 'date']).reset_index(drop=True)
df['price_range'] = (df['high'] - df['low']) / df['open']
df['open_close_change'] = (df['close'] - df['open']) / df['open']
df['prev_percent_change_price'] = df.groupby('stock')['percent_change_price'].shift(1)
df['volume_change_ratio'] = df['volume'] / df['previous_weeks_volume']
df = df.drop(columns=['open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close'])
# print(df.isnull().sum())

df = df[df['prev_percent_change_price'].notnull()].reset_index(drop=True)
# print(df.isnull().sum())
# print(df.shape)

X = df.drop(columns=['percent_change_next_weeks_price', 'date'])
y = df['percent_change_next_weeks_price']
X = pd.get_dummies(X, columns=['stock'], drop_first=True)
# print(X.shape)

X_train = X[df['quarter'] == 1]
X_test = X[df['quarter'] == 2]
y_train = y[df['quarter'] == 1]
y_test = y[df['quarter'] == 2]
# print(X_train.shape, X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# models = {
#     'Linear Regression': (LinearRegression(), X_train_scaled, X_test_scaled),
#     'Random Forest': (RandomForestRegressor(random_state=42), X_train, X_test),
#     'Gradient Boosting': (GradientBoostingRegressor(random_state=42), X_train, X_test)
# }
#
# results = []
# for name, (model, X_tr, X_te) in models.items():
#     model.fit(X_tr, y_train)
#     y_pred = model.predict(X_te)
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = mean_squared_error(y_test, y_pred) ** 0.5
#     results.append({'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse})
# results_df = pd.DataFrame(results)
# # print(results_df)

from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# tscv = TimeSeriesSplit(n_splits=5)
# cv_results = []
#
# for name, (model, X_tr, X_te) in models.items():
#     scores = cross_val_score(model, X_tr, y_train, cv=tscv, scoring='r2')
#     cv_results.append({
#         'Model': name,
#         'CV_R2_Mean': scores.mean(),
#         'CV_R2_Std': scores.std(),
#         'CV_R2_Scores': scores
#     })
# cv_results_df = pd.DataFrame(cv_results)
# print(cv_results_df[['Model', 'CV_R2_Mean', 'CV_R2_Std']])

X_v2 = df.drop(columns=['percent_change_next_weeks_price', 'date', 'stock'])
X_train_v2 = X_v2[df['quarter'] == 1]
X_test_v2 = X_v2[df['quarter'] == 2]
# print(X_train_v2.shape)

scaler_v2 = StandardScaler()
X_train_v2_scaled = scaler_v2.fit_transform(X_train_v2)
X_test_v2_scaled = scaler_v2.transform(X_test_v2)
X_train_v2_scaled = pd.DataFrame(X_train_v2_scaled, columns=X_train_v2.columns, index=X_train_v2.index)
X_test_v2_scaled = pd.DataFrame(X_test_v2_scaled, columns=X_test_v2.columns, index=X_test_v2.index)

from sklearn.linear_model import Ridge, Lasso

models_v2 = {
    # 'Ridge': (Ridge(alpha=10.0, random_state=42), X_train_v2_scaled, X_test_v2_scaled),
    # 'Lasso': (Lasso(alpha=0.1, random_state=42), X_train_v2_scaled, X_test_v2_scaled),
    'Random Forest (tuned)': (RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=42), X_train_v2, X_test_v2),
    # 'Gradient Boosting (tuned)': (GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, min_samples_leaf=10, random_state=42), X_train_v2, X_test_v2)
}
# cv_results_v2 = []
# tscv = TimeSeriesSplit(n_splits=5)
# for name, (model, X_tr, X_te) in models_v2.items():
#     scores = cross_val_score(model, X_tr, y_train, cv=tscv, scoring='r2')
#     cv_results_v2.append({
#         'Model': name,
#         'CV_R2_Mean': scores.mean(),
#         'CV_R2_Std': scores.std()
#     })
# cv_results_v2_df = pd.DataFrame(cv_results_v2)
# print(cv_results_v2_df)

final_model = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=42)
final_model.fit(X_train_v2, y_train)
y_pred_final = final_model.predict(X_test_v2)
r2_final = r2_score(y_test, y_pred_final)
mae_final = mean_absolute_error(y_test, y_pred_final)
rmse_final = mean_squared_error(y_test, y_pred_final) ** 0.5
# print(f"R2: {r2_final:.4f}")
# print(f"MAE: {mae_final:.4f}")
# print(f"RMSE: {rmse_final:.4f}")

# importance_df = pd.DataFrame({
#     'Feature': X_train_v2.columns,
#     'Importance': final_model.feature_importances_
# }).sort_values('Importance', ascending=False)
# print(importance_df)
#
# plt.figure(figsize=(10, 6))
# sns.barplot(data=importance_df, x='Importance', y='Feature')
# plt.title('Feature Importance - Random Forest')
# plt.tight_layout()
# plt.show()

import joblib

joblib.dump(final_model, 'ModelsOutcome/dow_jones_rf_model.pkl')
joblib.dump(list(X_train_v2.columns), 'ModelsOutcome/dow_jones_feature_columns.pkl')
joblib.dump(scaler_v2, 'ModelsOutcome/dow_jones_scaler.pkl')
print('model saved')
