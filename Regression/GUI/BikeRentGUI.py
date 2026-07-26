import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# رنگ هدر ثابت (هماهنگ با استایل پروژه‌های دیگه)
# --------------------------------------------------------------------
HEADER_COLOR = "#1F4E79"

# --------------------------------------------------------------------
# نگاشت مقادیر خوانا به کدهای عددی که دیتاست/مدل انتظار داره
# --------------------------------------------------------------------
SEASON_MAP = {"بهار": 1, "تابستان": 2, "پاییز": 3, "زمستان": 4}
YEAR_MAP = {"۲۰۱۱": 0, "۲۰۱۲": 1}
WEEKDAY_MAP = {
    "شنبه": 0, "یکشنبه": 1, "دوشنبه": 2, "سه‌شنبه": 3,
    "چهارشنبه": 4, "پنج‌شنبه": 5, "جمعه": 6,
}
WEATHER_MAP = {
    "صاف / کمی ابری": 1,
    "مه / ابری": 2,
    "برف یا باران سبک": 3,
    "باران / برف شدید": 4,
}
YES_NO_MAP = {"خیر": 0, "بله": 1}


# --------------------------------------------------------------------
# همون تابع feature engineering که در آموزش مدل استفاده شد.
# باید دقیقاً یکسان باشه، وگرنه ورودی مدل با چیزی که train شده فرق می‌کنه.
# --------------------------------------------------------------------
def engineer_features(df, is_hourly=True):
    df = df.copy()

    cols_to_drop = ['instant', 'dteday', 'casual', 'registered', 'temp']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
    df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df = df.drop(columns=['mnth', 'weekday'])

    if is_hourly and 'hr' in df.columns:
        df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
        df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24)
        df = df.drop(columns=['hr'])

    df['season'] = pd.Categorical(df['season'], categories=[1, 2, 3, 4])
    df['weathersit'] = pd.Categorical(df['weathersit'], categories=[1, 2, 3, 4])
    df = pd.get_dummies(df, columns=['season', 'weathersit'],
                         prefix=['season', 'weather'], drop_first=True)

    return df


class BikeRentPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("پیش‌بینی تعداد اجاره دوچرخه")

        # اندازه‌ی پنجره بر اساس اندازه صفحه کاربر (نه مقدار ثابت)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width, height = 620, 620
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(560, 600)

        self.model_package = self._load_model()

        self._build_ui()

    # ------------------------------------------------------------
    def _load_model(self):
        try:
            package = joblib.load('../ModelsOutcome/bike_rental_model.pkl')
            return package
        except FileNotFoundError:
            messagebox.showerror(
                "خطا",
                "فایل مدل (bike_rental_model.pkl) پیدا نشد.\n"
                "اول اسکریپت BikeRent_train_and_save.py را اجرا کن."
            )
            self.root.destroy()
            raise

    # ------------------------------------------------------------
    def _build_ui(self):
        header = tk.Label(
            self.root, text="پیش‌بینی تعداد اجاره دوچرخه (ساعتی)",
            bg=HEADER_COLOR, fg="white", font=("Tahoma", 14, "bold"),
            pady=12
        )
        header.pack(fill="x")

        container = ttk.Frame(self.root, padding=15)
        container.pack(fill="both", expand=True)

        # ---------------- فریم زمان ----------------
        time_frame = ttk.LabelFrame(container, text="اطلاعات زمانی", padding=10)
        time_frame.pack(fill="x", pady=(0, 10))

        self.season_cb = self._add_combobox(time_frame, "فصل:", list(SEASON_MAP.keys()), 0)
        self.year_cb = self._add_combobox(time_frame, "سال:", list(YEAR_MAP.keys()), 1)
        self.month_cb = self._add_combobox(time_frame, "ماه (1 تا 12):", [str(i) for i in range(1, 13)], 2)
        self.hour_cb = self._add_combobox(time_frame, "ساعت (0 تا 23):", [str(i) for i in range(0, 24)], 3)
        self.weekday_cb = self._add_combobox(time_frame, "روز هفته:", list(WEEKDAY_MAP.keys()), 4)
        self.holiday_cb = self._add_combobox(time_frame, "تعطیل رسمی است؟", list(YES_NO_MAP.keys()), 5)
        self.workingday_cb = self._add_combobox(time_frame, "روز کاری است؟", list(YES_NO_MAP.keys()), 6)

        for cb in (self.season_cb, self.year_cb, self.month_cb, self.hour_cb,
                   self.weekday_cb, self.holiday_cb, self.workingday_cb):
            cb.current(0)

        # ---------------- فریم آب‌وهوا ----------------
        weather_frame = ttk.LabelFrame(container, text="اطلاعات آب‌وهوایی", padding=10)
        weather_frame.pack(fill="x", pady=(0, 10))

        self.weather_cb = self._add_combobox(weather_frame, "وضعیت آب‌وهوا:", list(WEATHER_MAP.keys()), 0)
        self.weather_cb.current(0)

        self.atemp_entry = self._add_entry(weather_frame, "دمای احساسی (بین 0 و 1):", 1, default="0.5")
        self.hum_entry = self._add_entry(weather_frame, "رطوبت (بین 0 و 1):", 2, default="0.5")
        self.windspeed_entry = self._add_entry(weather_frame, "سرعت باد (بین 0 و 1):", 3, default="0.2")

        # ---------------- دکمه و نتیجه ----------------
        action_frame = ttk.Frame(container)
        action_frame.pack(fill="x", pady=(10, 0))

        predict_btn = tk.Button(
            action_frame, text="پیش‌بینی کن", bg=HEADER_COLOR, fg="white",
            font=("Tahoma", 11, "bold"), command=self.btn_predict_click
        )
        predict_btn.pack(pady=(0, 10), fill="x")

        self.result_label = tk.Label(
            action_frame, text="نتیجه اینجا نمایش داده می‌شود",
            font=("Tahoma", 13, "bold"), fg=HEADER_COLOR
        )
        self.result_label.pack()

    # ------------------------------------------------------------
    def _add_combobox(self, parent, label_text, values, row):
        lbl = ttk.Label(parent, text=label_text)
        lbl.grid(row=row, column=1, sticky="e", padx=5, pady=4)
        cb = ttk.Combobox(parent, values=values, state="readonly", width=25)
        cb.grid(row=row, column=0, sticky="w", padx=5, pady=4)
        return cb

    def _add_entry(self, parent, label_text, row, default=""):
        lbl = ttk.Label(parent, text=label_text)
        lbl.grid(row=row, column=1, sticky="e", padx=5, pady=4)
        entry = ttk.Entry(parent, width=27)
        entry.insert(0, default)
        entry.grid(row=row, column=0, sticky="w", padx=5, pady=4)
        return entry

    # ------------------------------------------------------------
    def _validate_normalized(self, value, field_name):
        try:
            v = float(value)
        except ValueError:
            raise ValueError(f"مقدار «{field_name}» باید یک عدد باشد.")
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"مقدار «{field_name}» باید بین 0 و 1 باشد.")
        return v

    # ------------------------------------------------------------
    def btn_predict_click(self):
        try:
            atemp = self._validate_normalized(self.atemp_entry.get(), "دمای احساسی")
            hum = self._validate_normalized(self.hum_entry.get(), "رطوبت")
            windspeed = self._validate_normalized(self.windspeed_entry.get(), "سرعت باد")
        except ValueError as e:
            messagebox.showwarning("مقدار نامعتبر", str(e))
            return

        raw_row = {
            'season': SEASON_MAP[self.season_cb.get()],
            'yr': YEAR_MAP[self.year_cb.get()],
            'mnth': int(self.month_cb.get()),
            'hr': int(self.hour_cb.get()),
            'holiday': YES_NO_MAP[self.holiday_cb.get()],
            'weekday': WEEKDAY_MAP[self.weekday_cb.get()],
            'workingday': YES_NO_MAP[self.workingday_cb.get()],
            'weathersit': WEATHER_MAP[self.weather_cb.get()],
            'atemp': atemp,
            'hum': hum,
            'windspeed': windspeed,
        }

        raw_df = pd.DataFrame([raw_row])
        features_df = engineer_features(raw_df, is_hourly=True)

        # هماهنگ کردن دقیق ستون‌ها با چیزی که مدل موقع train دیده
        features_df = features_df.reindex(
            columns=self.model_package['feature_names'], fill_value=0
        )

        prediction = self.model_package['model'].predict(features_df)[0]
        prediction = max(0, round(prediction))

        self.result_label.config(
            text=f"تعداد اجاره پیش‌بینی‌شده: {prediction} دوچرخه"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = BikeRentPredictorApp(root)
    root.mainloop()