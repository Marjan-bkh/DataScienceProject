# import joblib
# features = joblib.load("../Regression/ModelsOutcome/bike_rental_model.pkl")
# print(features)

import numpy as np


def transform_input(yr, holiday, workingday, atemp, hum, windspeed, mnth, weekday, hr, season, weather):
    # ۱. فیچرهای چرخه‌ای (sin/cos) رو می‌سازیم
    mnth_sin = np.sin(2 * np.pi * mnth / 12)
    mnth_cos = np.cos(2 * np.pi * mnth / 12)

    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)

    hr_sin = np.sin(2 * np.pi * hr / 24)
    hr_cos = np.cos(2 * np.pi * hr / 24)

    # ۲. one-hot برای season (بهار=پایه، پس اگه season==1 همه صفرن)
    season_2 = 1 if season == 2 else 0
    season_3 = 1 if season == 3 else 0
    season_4 = 1 if season == 4 else 0

    # ۳. one-hot برای weather (صاف=پایه، پس اگه weather==1 همه صفرن)
    weather_2 = 1 if weather == 2 else 0
    weather_3 = 1 if weather == 3 else 0
    weather_4 = 1 if weather == 4 else 0

    # ۴. دیکشنری نهایی، دقیقاً به همون ترتیب feature_names
    final_input = {
        'yr': yr,
        'holiday': holiday,
        'workingday': workingday,
        'atemp': atemp,
        'hum': hum,
        'windspeed': windspeed,
        'mnth_sin': mnth_sin,
        'mnth_cos': mnth_cos,
        'weekday_sin': weekday_sin,
        'weekday_cos': weekday_cos,
        'hr_sin': hr_sin,
        'hr_cos': hr_cos,
        'season_2': season_2,
        'season_3': season_3,
        'season_4': season_4,
        'weather_2': weather_2,
        'weather_3': weather_3,
        'weather_4': weather_4,
    }

    return final_input

import joblib
import pandas as pd

saved = joblib.load("../Regression/ModelsOutcome/bike_rental_model.pkl")
model = saved['model']
feature_names = saved['feature_names']

sample = transform_input(
    yr=1, holiday=0, workingday=1, atemp=25.0, hum=50.0, windspeed=10.0,
    mnth=7, weekday=3, hr=14, season=2, weather=1
)

input_df = pd.DataFrame([sample])
input_df = input_df[feature_names]  # مطمئن میشیم ترتیب ستون‌ها دقیقاً درسته

prediction = model.predict(input_df)
print(prediction)