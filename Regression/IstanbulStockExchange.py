import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

df= pd.read_excel('DataSets/data_akbilgic.xlsx',header=1)
# print(df.head())
# print(df.describe())
# print(df.shape)
# print(df.columns.values)
columns = df.columns.values.tolist()
features = ['ISE', 'SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM']
#
# fig,ax = plt.subplots(2,4,figsize=(10,10))
# axes =ax.flatten()
# for i, col in enumerate(features):
#     axes[i].hist(df[col],bins=10,color='blue')
#     axes[i].set_title(col)
# plt.tight_layout()
# plt.show()

numeric_cols = df.columns.drop('date')
# stats = pd.DataFrame({
#     'skewness': df[numeric_cols].skew(),
#     'kurtosis': df[numeric_cols].kurtosis()
# })
# print(stats)

# df[numeric_cols].plot(kind='box', subplots=True, layout=(2,5), figsize=(16,8))
# plt.tight_layout()
# plt.show()

# corr_matrix = df[numeric_cols].corr()
# print(corr_matrix.to_string())
#
# plt.figure(figsize=(10,10))
# sns.heatmap(corr_matrix, annot=True, cmap="YlOrRd")
# plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor

# # فیچرهایی که می‌خوایم برای پیش‌بینی ISE استفاده کنیم
# feature_cols = ['SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM']
# X = df[feature_cols]
# vif_data = pd.DataFrame()
# vif_data['feature'] = X.columns
# vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# print(vif_data)

# feature_cols_v2 = ['SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EM']
# X2 = df[feature_cols_v2]
# vif_data_v2 = pd.DataFrame()
# vif_data_v2['feature'] = X2.columns
# vif_data_v2['VIF'] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
# print(vif_data_v2)

from sklearn.model_selection import train_test_split

X = df[['SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EM']]
y = df['ISE']

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
# print(X_train.shape, X_test.shape)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}
tscv = TimeSeriesSplit(n_splits=5)
results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
    results.append({
        'model': name,
        'mean_R2': scores.mean(),
        'std_R2': scores.std(),
        'mean_minus_std': scores.mean() - scores.std()
    })
results_df = pd.DataFrame(results).sort_values('mean_minus_std', ascending=False)
# print(results_df)

# محاسبه‌ی R² معتبر با CV (روی همون split زمانی که قبلاً استفاده کردیم)
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(LinearRegression(), X, y, cv=tscv, scoring='r2')
reported_r2 = cv_scores.mean()




feature_cols = ['SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EM']
X = df[feature_cols]
y = df['ISE']

final_model = LinearRegression()
final_model.fit(X, y)

import joblib

joblib.dump(
    {'model': final_model, 'feature_names': feature_cols},
    'ModelsOutcome/istanbul_stock_linear_regression_model.pkl' )
# ذخیره برای استفاده تو UI
joblib.dump(reported_r2,  'ModelsOutcome/istanbul_stock_model_r2.pkl')


print("model saved")
# print(f"ضرایب مدل:")
# for feat, coef in zip(feature_cols, final_model.coef_):
#     print(f"  {feat}: {coef:.4f}")
# print(f"Intercept: {final_model.intercept_:.5f}")