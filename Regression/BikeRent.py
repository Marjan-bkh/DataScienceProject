import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Regression.ConcreteStrength import X_train

# تنظیمات نمایش بهتر
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

df_day = pd.read_csv('DataSets/BikeRent_day.csv')
df_hour = pd.read_csv('DataSets/BikeRent_hour.csv')

# print("=== day.csv ===")
# print(df_day.shape)
# print(df_day.info())
# print(df_day.head())
#
# print("\n=== hour.csv ===")
# print(df_hour.shape)
# print(df_hour.info())
# print(df_hour.head())
# print(df_hour.columns.values)
# print(df_day.columns.values)

# print(df_day.describe())
# print(df_hour.describe())
# پیدا کردن رکوردهای رطوبت صفر
# zero_hum = df_hour[df_hour['hum'] == 0]
# print(f"تعداد رکوردهای hum=0: {len(zero_hum)}")
# print(zero_hum[['dteday', 'hr', 'hum', 'temp', 'weathersit', 'cnt']])
#
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
#
# axes[0,0].hist(df_hour['cnt'], bins=50, color='steelblue')
# axes[0,0].set_title('توزیع cnt (hour.csv)')
#
# axes[0,1].hist(df_hour['temp'], bins=50, color='orange')
# axes[0,1].set_title('توزیع temp')
#
# axes[0,2].hist(df_hour['hum'], bins=50, color='green')
# axes[0,2].set_title('توزیع hum')
#
# axes[1,0].hist(df_hour['windspeed'], bins=50, color='purple')
# axes[1,0].set_title('توزیع windspeed')
#
# axes[1,1].hist(df_hour['casual'], bins=50, color='red')
# axes[1,1].set_title('توزیع casual')
#
# axes[1,2].hist(df_hour['registered'], bins=50, color='brown')
# axes[1,2].set_title('توزیع registered')
#
# plt.tight_layout()
# plt.show()
#
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
# axes[0].boxplot(df_hour['cnt'])
# axes[0].set_title('Boxplot cnt (hour.csv)')
#
# axes[1].boxplot(df_day['cnt'])
# axes[1].set_title('Boxplot cnt (day.csv)')
#
# plt.tight_layout()
# plt.show()
#
# numeric_cols = ['temp', 'atemp', 'hum', 'windspeed', 'season', 'yr', 'mnth',
#                  'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'cnt']
#
# corr_matrix = df_hour[numeric_cols].corr()
#
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
# plt.title('همبستگی بین متغیرها (hour.csv)')
# plt.tight_layout()
# plt.show()

df_hour_clean = df_hour[df_hour['hum'] != 0].copy()
print(f"تعداد رکوردها قبل از حذف: {len(df_hour)}")
print(f"تعداد رکوردها بعد از حذف: {len(df_hour_clean)}")

df_hour_clean['dteday'] = pd.to_datetime(df_hour_clean['dteday'])
df_day['dteday'] = pd.to_datetime(df_day['dteday'])

print(df_hour_clean['dteday'].min(), "تا", df_hour_clean['dteday'].max())
print(df_day['dteday'].min(), "تا", df_day['dteday'].max())
# پیدا کردن تاریخ مناسب برای تقسیم train, test
cutoff_date = df_hour_clean['dteday'].quantile(0.8, interpolation='nearest')
print(f"تاریخ برش: {cutoff_date}")

train_hour = df_hour_clean[df_hour_clean['dteday'] < cutoff_date].copy()
test_hour = df_hour_clean[df_hour_clean['dteday'] >= cutoff_date].copy()
print(f"Train: {len(train_hour)} رکورد ({len(train_hour)/len(df_hour_clean)*100:.1f}%)")
print(f"Test:  {len(test_hour)} رکورد ({len(test_hour)/len(df_hour_clean)*100:.1f}%)")
print(f"بازهٔ Train: {train_hour['dteday'].min()} تا {train_hour['dteday'].max()}")
print(f"بازهٔ Test:  {test_hour['dteday'].min()} تا {test_hour['dteday'].max()}")

train_day = df_day[df_day['dteday'] < cutoff_date].copy()
test_day = df_day[df_day['dteday'] >= cutoff_date].copy()
print(f"\nTrain day: {len(train_day)} رکورد")
print(f"Test day:  {len(test_day)} رکورد")


def engineer_features(df, is_hourly=True):
    df = df.copy()

    # حذف ستون‌های بی‌فایده یا خطرناک
    cols_to_drop = ['instant', 'dteday', 'casual', 'registered', 'temp']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Cyclical encoding برای mnth (دوره=12) و weekday (دوره=7)
    df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
    df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df = df.drop(columns=['mnth', 'weekday'])

    # Cyclical encoding برای hr (فقط تو hour.csv وجود داره)
    if is_hourly and 'hr' in df.columns:
        df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
        df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24)
        df = df.drop(columns=['hr'])
        # مشخص کردن دستی همهٔ سطوح ممکن، تا Train و Test همیشه ستون‌های یکسان بسازن
        df['season'] = pd.Categorical(df['season'], categories=[1, 2, 3, 4])
        df['weathersit'] = pd.Categorical(df['weathersit'], categories=[1, 2, 3, 4])
    # One-Hot Encoding برای season و weathersit
    df = pd.get_dummies(df, columns=['season', 'weathersit'], prefix=['season', 'weather'], drop_first=True)

    return df


# اعمال روی هر چهار دیتافریم
train_hour_fe = engineer_features(train_hour, is_hourly=True)
test_hour_fe = engineer_features(test_hour, is_hourly=True)
train_day_fe = engineer_features(train_day, is_hourly=False)
test_day_fe = engineer_features(test_day, is_hourly=False)

# print("ستون‌های نهایی train_hour_fe:")
# print(train_hour_fe.columns.tolist())
# print(f"\nShape: {train_hour_fe.shape}")
# print(train_hour_fe.head())

X_train_hour = train_hour_fe.drop(columns=['cnt'])
Y_train_hour = train_hour_fe['cnt']
X_test_hour = test_hour_fe.drop(columns=['cnt'])
Y_test_hour = test_hour_fe['cnt']
X_test_hour = X_test_hour.reindex(columns=X_train_hour.columns, fill_value=0)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"--- {model_name} ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.4f}")
    print()

    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'y_pred': y_pred}

#Hour csv
results_hour = []

lr = LinearRegression()
results_hour.append(evaluate_model(lr, X_train_hour, Y_train_hour, X_test_hour, Y_test_hour, "Linear Regression"))

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
results_hour.append(evaluate_model(rf, X_train_hour, Y_train_hour, X_test_hour, Y_test_hour, "Random Forest"))

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
results_hour.append(evaluate_model(gb, X_train_hour, Y_train_hour, X_test_hour, Y_test_hour, "Gradient Boosting"))

#Day csv
X_train_day = train_day_fe.drop(columns=['cnt'])
y_train_day = train_day_fe['cnt']

X_test_day = test_day_fe.drop(columns=['cnt'])
y_test_day = test_day_fe['cnt']
X_test_day = X_test_day.reindex(columns=X_train_day.columns, fill_value=0)


print(f"X_train_day shape: {X_train_day.shape}")
print(f"ستون‌های X: {X_train_day.columns.tolist()}")

results_day = []

lr_day = LinearRegression()
results_day.append(evaluate_model(lr_day, X_train_day, y_train_day, X_test_day, y_test_day, "Linear Regression (day)"))

rf_day = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
results_day.append(evaluate_model(rf_day, X_train_day, y_train_day, X_test_day, y_test_day, "Random Forest (day)"))

gb_day = GradientBoostingRegressor(n_estimators=100, random_state=42)
results_day.append(evaluate_model(gb_day, X_train_day, y_train_day, X_test_day, y_test_day, "Gradient Boosting (day)"))

from sklearn.model_selection import TimeSeriesSplit, cross_val_score


def run_cv(X, y, model, model_name, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')

    print(f"--- {model_name} ---")
    print(f"R² هر فولد: {np.round(scores, 4)}")
    print(f"میانگین R²: {scores.mean():.4f}")
    print(f"انحراف‌معیار R²: {scores.std():.4f}")
    print()

    return scores

print("=========== Cross-Validation روی hour.csv ===========\n")
run_cv(X_train_hour, Y_train_hour, LinearRegression(), "Linear Regression (hour)")
run_cv(X_train_hour, Y_train_hour, RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), "Random Forest (hour)")
run_cv(X_train_hour, Y_train_hour, GradientBoostingRegressor(n_estimators=100, random_state=42), "Gradient Boosting (hour)")

print("=========== Cross-Validation روی day.csv ===========\n")
run_cv(X_train_day, y_train_day, LinearRegression(), "Linear Regression (day)")
run_cv(X_train_day, y_train_day, RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), "Random Forest (day)")
run_cv(X_train_day, y_train_day, GradientBoostingRegressor(n_estimators=100, random_state=42), "Gradient Boosting (day)")

rf_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_final.fit(X_train_hour, Y_train_hour)

importances = pd.DataFrame({
    'feature': X_train_hour.columns,
    'importance': rf_final.feature_importances_
}).sort_values('importance', ascending=False)

print(importances)

plt.figure(figsize=(10, 8))
plt.barh(importances['feature'], importances['importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest (hour.csv)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

from sklearn.inspection import permutation_importance

perm_result = permutation_importance(rf_final, X_test_hour, Y_test_hour, n_repeats=10, random_state=42, n_jobs=-1)

perm_importances = pd.DataFrame({
    'feature': X_test_hour.columns,
    'importance_mean': perm_result.importances_mean,
    'importance_std': perm_result.importances_std
}).sort_values('importance_mean', ascending=False)

print(perm_importances)

