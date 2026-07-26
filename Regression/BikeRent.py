import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor

# ==========================================================================
# این نسخه فقط شامل مراحل ضروری برای ساخت و ذخیره‌ی مدل نهاییه.
# مراحل اکتشافی (EDA، نمودارها، مقایسه مدل‌ها، CV، day.csv و ...) که قبلاً
# انجام‌شون دادیم و به یک نتیجه رسیدیم (Random Forest روی hour.csv)،
# پایین همین فایل به‌صورت کامنت نگه داشته شدن تا اگه لازم شد دوباره
# قابل اجرا باشن، ولی برای اجرای روتین "train و ذخیره مدل" لازم نیستن.
# ==========================================================================

df_hour = pd.read_csv('DataSets/BikeRent_hour.csv')

# --------------------------------------------------------------------
# پاکسازی: حذف ۲۲ رکورد خراب hum=0 (خرابی سنسور در 2011-03-10)
# --------------------------------------------------------------------
df_hour_clean = df_hour[df_hour['hum'] != 0].copy()
print(f"تعداد رکوردها قبل از حذف: {len(df_hour)}")
print(f"تعداد رکوردها بعد از حذف: {len(df_hour_clean)}")


# --------------------------------------------------------------------
# Feature Engineering (همون تابعی که قبلاً ساختیم)
# --------------------------------------------------------------------
def engineer_features(df, is_hourly=True):
    df = df.copy()

    # حذف ستون‌های بی‌فایده یا خطرناک (نشتی داده)
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

    # مشخص کردن دستی همه‌ی سطوح ممکن، تا Train/Test/داده جدید همیشه
    # ستون‌های dummy یکسانی بسازن (حل مشکل ValueError قبلی)
    df['season'] = pd.Categorical(df['season'], categories=[1, 2, 3, 4])
    df['weathersit'] = pd.Categorical(df['weathersit'], categories=[1, 2, 3, 4])

    df = pd.get_dummies(df, columns=['season', 'weathersit'],
                         prefix=['season', 'weather'], drop_first=True)

    return df


df_hour_fe = engineer_features(df_hour_clean, is_hourly=True)

X_full_hour = df_hour_fe.drop(columns=['cnt'])
y_full_hour = df_hour_fe['cnt']

print(f"تعداد فیچرها: {X_full_hour.shape[1]}")
print(f"ستون‌های مدل: {list(X_full_hour.columns)}")

# --------------------------------------------------------------------
# Train نهایی روی کل داده (چون قبلاً با Train/Test و CV ارزیابی‌اش
# کردیم و از عملکردش مطمئن شدیم؛ الان از تمام داده برای دقت بیشتر
# استفاده می‌کنیم)
# --------------------------------------------------------------------
final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
final_model.fit(X_full_hour, y_full_hour)

# --------------------------------------------------------------------
# ذخیره مدل + متادیتای لازم (تا در GUI بدونیم دقیقاً چه فیچرهایی و
# به چه ترتیبی باید به مدل بدیم)
# --------------------------------------------------------------------
model_package = {
    'model': final_model,
    'feature_names': list(X_full_hour.columns),
    'model_type': 'RandomForestRegressor',
    'trained_on': 'hour.csv (Capital Bikeshare 2011-2012)',
    'performance_cv_r2_mean': 0.7174,  # از نتایج Cross-Validation قبلی
}

joblib.dump(model_package, 'ModelsOutcome/bike_rental_model.pkl')
print("مدل با موفقیت ذخیره شد: bike_rental_model.pkl")

# تست سریع
sample = X_full_hour.iloc[[0]]
prediction = final_model.predict(sample)
print(f"پیش‌بینی برای نمونه اول: {prediction[0]:.1f}  |  مقدار واقعی: {y_full_hour.iloc[0]}")


# ==========================================================================
#                     مراحل اکتشافی/تحلیلی قبلی (آرشیو)
#   این بخش‌ها برای رسیدن به نتیجه‌ی بالا لازم بودن ولی برای اجرای
#   روتین train+save ضروری نیستن. اگه خواستی EDA/CV/مقایسه مدل‌ها رو
#   دوباره ببینی، کامنت‌شون رو بردار.
# ==========================================================================

# --- EDA (نمودارها، describe، heatmap و ...) ---
# import matplotlib.pyplot as plt
# import seaborn as sns
# pd.set_option('display.max_columns', None)
# sns.set_style('whitegrid')
# ... (کدهای EDA که قبلاً نوشتیم)

# --- Train/Test split زمانی + مقایسه مدل‌ها (LR / RF / GB) ---
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# ... (کد split + evaluate_model که قبلاً نوشتیم)

# --- day.csv ---
# نتیجه‌گیری قبلی: day.csv فقط 731 رکورد داره و با TimeSeriesSplit
# نتایج بسیار ناپایدار و حتی R² منفی گرفتیم (overfitting شدید روی
# داده کم). به همین دلیل day.csv کاملاً از پروژه کنار گذاشته شد.

# --- Cross-Validation با TimeSeriesSplit ---
# from sklearn.model_selection import TimeSeriesSplit, cross_val_score
# ... (کد run_cv که قبلاً نوشتیم)

# --- Feature Importance + Permutation Importance ---
# from sklearn.inspection import permutation_importance
# ... (کدهای importances / perm_importances که قبلاً نوشتیم)