import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv('DataSets/Daily_Demand_Forecasting_Orders.csv', sep=';')
# print(df.shape)       # (60, 13)
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# print(df.columns.values)

# # هیت‌مپ همبستگی
# plt.figure(figsize=(12, 8))
# sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()
#
# # توزیع target
# sns.histplot(df['Target (Total orders)'], kde=True)
# plt.show()

# Delete columns to avoid data leakage
cols_to_drop = [
    'Non-urgent order',
    'Urgent order',
    'Order type A',
    'Order type B',
    'Order type C',
    'Target (Total orders)'
]

x = df.drop(columns=cols_to_drop)
y = df['Target (Total orders)']

#Normalization
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor

models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'RandomForestRegressor': RandomForestRegressor()
        }
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name}: RMSE={rmse:.2f}, R²={r2:.4f}, MAE={mae:.2f}, MSE={mse:.2f}")

# مقایسه پیش‌بینی vs واقعی
# plt.scatter(y_test, y_pred, alpha=0.7)
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Actual vs Predicted')
# # create red line
# plt.plot([y_test.min(), y_test.max()],
#          [y_test.min(), y_test.max()], 'r--')
# plt.show()

from sklearn.model_selection import cross_val_score
# model= LinearRegression()
# scores = cross_val_score(model, x_scaled, y,cv=5, scoring='r2')
#
# print(f"R² در هر fold: {scores.round(3)}")
# print(f"میانگین R²: {scores.mean():.3f}")
# print(f"انحراف معیار: {scores.std():.3f}")

print("\n--- Cross Validation (5-Fold) ---")
for name, model in models.items():
    scores = cross_val_score(model, x_scaled, y, cv=5, scoring='r2')
    print(f"{name}: میانگین R²={scores.mean():.3f}, انحراف معیار={scores.std():.3f}")


# بهترین مدل رو بر اساس cross validation انتخاب کن
best_name = None
best_score = -999

for name, model in models.items():
    score = cross_val_score(model, x_scaled, y, cv=5, scoring='r2').mean()
    if score > best_score:
        best_score = score
        best_name = name
best_model = models[best_name]
best_model.fit(x_scaled, y)

print(f"\nبهترین مدل: {best_name}")



# ذخیره مدل و scaler
# joblib.dump(best_model, 'best_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# print("مدل ذخیره شد.")



