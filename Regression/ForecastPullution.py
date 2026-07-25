import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression , Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# DEWP (شبنم) , TEMP : دما , PRES: فشار هوا , cbwd : جهت باد, IWS: سرعت باد

df = pd.read_csv('DataSets/forecast_pullution.csv')
# df.head()
# df.info()
# df.describe()
# print(df.isnull().sum())
# print(df.columns.values)

#Delete rows that pm2.5 is null
df = df.dropna(subset=['pm2.5'])

df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df = df.sort_values("datetime").reset_index(drop=True)

#cycling encoding for hour and month-- close hour 0 to 23 and month 1 to 12
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

#one-hot encoding for categorical column for cbwd- jahate bad
df = pd.get_dummies(df, columns=["cbwd"], prefix="wd")
# print(df.columns)
wind_cols = [c for c in df.columns if c.startswith("wd_")]
# print(wind_cols)
feature_cols = [
                   "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir",
                   "hour_sin", "hour_cos", "month_sin", "month_cos",
               ] + wind_cols

#print(feature_cols)

X = df[feature_cols]
y = df["pm2.5"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
# print(f"Train: {X_train.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2, "model": model, "y_pred": y_pred}
    print(f"\n--- {name} ---")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R2:   {r2:.3f}")

comparison = pd.DataFrame({
    name: {"RMSE": res["RMSE"], "MAE": res["MAE"], "R2": res["R2"]}
    for name, res in results.items()
}).T
 # print(comparison)

best_model_name = comparison["R2"].idxmax()
# best_model_name = comparison.sort_values("R2",ascending= False).index[0]
print(f"\nبهترین مدل بر اساس R2: {best_model_name}")

best_pred = results[best_model_name]["y_pred"]

plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:500], label="واقعی", linewidth=1)
plt.plot(best_pred[:500], label="پیش‌بینی", linewidth=1, alpha=0.7)
plt.title(f"مقایسه مقادیر واقعی و پیش‌بینی‌شده ({best_model_name}) - ۵۰۰ نمونه اول تست")
plt.xlabel("نمونه")
plt.ylabel("PM2.5")
plt.legend()
plt.tight_layout()
# plt.savefig("prediction_vs_actual.png", dpi=120)
# print("\nنمودار در prediction_vs_actual.png ذخیره شد.")
#plt.show()

if "Random Forest" in results:
    rf_model = results["Random Forest"]["model"]
    #print(rf_model)
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    # print(importances)

    plt.figure(figsize=(10, 6))
    importances.plot(kind="bar")
    plt.title("اهمیت فیچرها (Random Forest)")
    plt.tight_layout()
    plt.show()
    # plt.savefig("feature_importance.png", dpi=120)
    # print("نمودار اهمیت فیچرها در feature_importance.png ذخیره شد.")