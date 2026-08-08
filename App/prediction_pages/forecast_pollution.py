import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np

WIND_DIR_MAP = {
    "Northeast": "wd_NE",
    "Northwest": "wd_NW",
    "Southeast": "wd_SE",
    "Calm/Variable": "wd_cv",
}


class PredictionPageForecastPollution(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="regression")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Beijing PM2.5 Air Pollution Forecast", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts PM2.5 air pollution concentration based on weather conditions "
            "and time features.\nModel: Random Forest Regressor."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (R²): 46.6%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Dew Point (°C):").grid(row=0, column=0, sticky="w", pady=5)
        self.dewp_entry = tk.Entry(form)
        self.dewp_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Temperature (°C):").grid(row=1, column=0, sticky="w", pady=5)
        self.temp_entry = tk.Entry(form)
        self.temp_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="Pressure (hPa):").grid(row=2, column=0, sticky="w", pady=5)
        self.pres_entry = tk.Entry(form)
        self.pres_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Cumulated Wind Speed (m/s):").grid(row=3, column=0, sticky="w", pady=5)
        self.iws_entry = tk.Entry(form)
        self.iws_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Cumulated Snow Hours:").grid(row=4, column=0, sticky="w", pady=5)
        self.is_entry = tk.Entry(form)
        self.is_entry.grid(row=4, column=1, pady=5)

        tk.Label(form, text="Cumulated Rain Hours:").grid(row=5, column=0, sticky="w", pady=5)
        self.ir_entry = tk.Entry(form)
        self.ir_entry.grid(row=5, column=1, pady=5)

        tk.Label(form, text="Hour (0-23):").grid(row=6, column=0, sticky="w", pady=5)
        self.hour_entry = tk.Entry(form)
        self.hour_entry.grid(row=6, column=1, pady=5)

        tk.Label(form, text="Month (1-12):").grid(row=7, column=0, sticky="w", pady=5)
        self.month_entry = tk.Entry(form)
        self.month_entry.grid(row=7, column=1, pady=5)

        tk.Label(form, text="Wind Direction:").grid(row=8, column=0, sticky="w", pady=5)
        self.wind_dir_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.wind_dir_var, values=list(WIND_DIR_MAP.keys()), state="readonly").grid(row=8, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            dewp = float(self.dewp_entry.get())
            temp = float(self.temp_entry.get())
            pres = float(self.pres_entry.get())
            iws = float(self.iws_entry.get())
            is_snow = float(self.is_entry.get())
            ir_rain = float(self.ir_entry.get())
            hour = int(self.hour_entry.get())
            month = int(self.month_entry.get())
            wind_col = WIND_DIR_MAP[self.wind_dir_var.get()]
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'DEWP': dewp,
            'TEMP': temp,
            'PRES': pres,
            'Iws': iws,
            'Is': is_snow,
            'Ir': ir_rain,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'wd_NE': 1 if wind_col == "wd_NE" else 0,
            'wd_NW': 1 if wind_col == "wd_NW" else 0,
            'wd_SE': 1 if wind_col == "wd_SE" else 0,
            'wd_cv': 1 if wind_col == "wd_cv" else 0,
        }

        model = joblib.load("../../Regression/ModelsOutcome/forecast_pollution_model.pkl")
        scaler = joblib.load("../../Regression/ModelsOutcome/forecast_pollution_scaler.pkl")
        feature_names = joblib.load("../../Regression/ModelsOutcome/forecast_pollution_features.pkl")

        input_df = pd.DataFrame([sample])[feature_names]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        self.result_label.config(text=f"Predicted PM2.5: {round(prediction, 1)} µg/m³")