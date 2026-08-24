import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

WEEKDAY_MAP = {
    "Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5, "Saturday": 6
}

YEAR_MAP = {"2011": 0, "2012": 1}
YES_NO_MAP = {"No": 0, "Yes": 1}
SEASON_MAP = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
WEATHER_MAP = {"Clear/Partly Cloudy": 1, "Mist/Cloudy": 2, "Light Rain/Snow": 3, "Heavy Rain/Snow": 4}


class PredictionPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="regression")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Bike Rental Demand Prediction", font=("Arial", 16, "bold")).pack(pady=10)
        description_text = (
            "Predicts hourly bike rental demand using weather, seasonal, and calendar data "
            "from the Capital Bikeshare system (2011-2012). \n Model: Random Forest Regressor."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(
            pady=(0, 5))

        tk.Label(self, text="Model Accuracy (Cross-Validated R²): 72%", font=("Arial", 10, "italic"), fg="gray8").pack(
            pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)


        tk.Label(form, text="Season:").grid(row=0, column=0, sticky="w", pady=5)
        self.season_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.season_var, values=list(SEASON_MAP.keys()), state="readonly").grid(row=0, column=1, pady=5)

        tk.Label(form, text="Weather:").grid(row=1, column=0, sticky="w", pady=5)
        self.weather_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.weather_var, values=list(WEATHER_MAP.keys()), state="readonly").grid(row=1, column=1, pady=5)

        tk.Label(form, text="Month:").grid(row=2, column=0, sticky="w", pady=5)
        self.mnth_var = tk.StringVar()

        ttk.Combobox(form, textvariable=self.mnth_var, values=list(MONTH_MAP.keys()), state="readonly").grid(row=2,column=1,pady=5)
        tk.Label(form, text="Day of Week:").grid(row=3, column=0, sticky="w", pady=5)
        self.weekday_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.weekday_var, values=list(WEEKDAY_MAP.keys()), state="readonly").grid(row=3,column=1,pady=5)

        tk.Label(form, text="Hour (0-23):").grid(row=4, column=0, sticky="w", pady=5)
        self.hr_entry = tk.Entry(form)
        self.hr_entry.grid(row=4, column=1, pady=5)

        tk.Label(form, text="Feels-like Temperature (°C):").grid(row=5, column=0, sticky="w", pady=5)
        self.atemp_entry = tk.Entry(form)
        self.atemp_entry.grid(row=5, column=1, pady=5)

        tk.Label(form, text="Humidity (%):").grid(row=6, column=0, sticky="w", pady=5)
        self.hum_entry = tk.Entry(form)
        self.hum_entry.grid(row=6, column=1, pady=5)

        tk.Label(form, text="Windspeed (km/h):").grid(row=7, column=0, sticky="w", pady=5)
        self.windspeed_entry = tk.Entry(form)
        self.windspeed_entry.grid(row=7, column=1, pady=5)

        tk.Label(form, text="Year:").grid(row=8, column=0, sticky="w", pady=5)
        self.yr_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.yr_var, values=list(YEAR_MAP.keys()), state="readonly").grid(row=8,column=1,pady=5)

        tk.Label(form, text="Holiday:").grid(row=9, column=0, sticky="w", pady=5)
        self.holiday_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.holiday_var, values=list(YES_NO_MAP.keys()), state="readonly").grid(row=9,column=1,pady=5)

        tk.Label(form, text="Working Day:").grid(row=10, column=0, sticky="w", pady=5)
        self.workingday_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.workingday_var, values=list(YES_NO_MAP.keys()), state="readonly").grid(row=10, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self, dataset=None):
        pass

    def on_predict(self):
        try:
            season = SEASON_MAP[self.season_var.get()]
            weather = WEATHER_MAP[self.weather_var.get()]
            mnth = MONTH_MAP[self.mnth_var.get()]
            weekday = WEEKDAY_MAP[self.weekday_var.get()]
            hr = int(self.hr_entry.get())
            atemp = float(self.atemp_entry.get())
            hum = float(self.hum_entry.get())
            windspeed = float(self.windspeed_entry.get())
            atemp = atemp / 50
            hum = hum / 100
            windspeed = windspeed / 67
            yr = YEAR_MAP[self.yr_var.get()]
            holiday = YES_NO_MAP[self.holiday_var.get()]
            workingday = YES_NO_MAP[self.workingday_var.get()]
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return


        mnth_sin = np.sin(2 * np.pi * mnth / 12)
        mnth_cos = np.cos(2 * np.pi * mnth / 12)
        weekday_sin = np.sin(2 * np.pi * weekday / 7)
        weekday_cos = np.cos(2 * np.pi * weekday / 7)
        hr_sin = np.sin(2 * np.pi * hr / 24)
        hr_cos = np.cos(2 * np.pi * hr / 24)

        sample = {
            'yr': yr, 'holiday': holiday, 'workingday': workingday,
            'atemp': atemp, 'hum': hum, 'windspeed': windspeed,
            'mnth_sin': mnth_sin, 'mnth_cos': mnth_cos,
            'weekday_sin': weekday_sin, 'weekday_cos': weekday_cos,
            'hr_sin': hr_sin, 'hr_cos': hr_cos,
            'season_2': 1 if season == 2 else 0,
            'season_3': 1 if season == 3 else 0,
            'season_4': 1 if season == 4 else 0,
            'weather_2': 1 if weather == 2 else 0,
            'weather_3': 1 if weather == 3 else 0,
            'weather_4': 1 if weather == 4 else 0,
        }

        saved = joblib.load(os.path.join(BASE_DIR, "Regression", "ModelsOutcome", "bike_rental_model.pkl"))
        model = saved['model']
        feature_names = saved['feature_names']

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        self.result_label.config(text=f"Predicted rentals: {round(prediction)}")

from App.category_page import CategoryPage