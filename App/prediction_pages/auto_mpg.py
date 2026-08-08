import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

ORIGIN_MAP = {"USA": 1, "Europe": 2, "Japan": 3}


class PredictionPageAutoMPG(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="regression")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Auto MPG Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts a car's fuel efficiency (miles per gallon) based on its technical "
            "specifications.\n Model: Linear Regression."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (R²): 82%", font=("Arial", 10, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Cylinders:").grid(row=0, column=0, sticky="w", pady=5)
        self.cylinders_entry = tk.Entry(form)
        self.cylinders_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Displacement (cu. in.):").grid(row=1, column=0, sticky="w", pady=5)
        self.displacement_entry = tk.Entry(form)
        self.displacement_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="Horsepower:").grid(row=2, column=0, sticky="w", pady=5)
        self.horsepower_entry = tk.Entry(form)
        self.horsepower_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Weight (lbs):").grid(row=3, column=0, sticky="w", pady=5)
        self.weight_entry = tk.Entry(form)
        self.weight_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Acceleration (0-60 time):").grid(row=4, column=0, sticky="w", pady=5)
        self.acceleration_entry = tk.Entry(form)
        self.acceleration_entry.grid(row=4, column=1, pady=5)

        tk.Label(form, text="Model Year (e.g. 70-82):").grid(row=5, column=0, sticky="w", pady=5)
        self.model_year_entry = tk.Entry(form)
        self.model_year_entry.grid(row=5, column=1, pady=5)

        tk.Label(form, text="Origin:").grid(row=6, column=0, sticky="w", pady=5)
        self.origin_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.origin_var, values=list(ORIGIN_MAP.keys()), state="readonly").grid(row=6, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            cylinders = int(self.cylinders_entry.get())
            displacement = float(self.displacement_entry.get())
            horsepower = float(self.horsepower_entry.get())
            weight = float(self.weight_entry.get())
            acceleration = float(self.acceleration_entry.get())
            model_year = int(self.model_year_entry.get())
            origin = ORIGIN_MAP[self.origin_var.get()]
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Input Error", "Please complete all fields.")
            return

        sample = {
            'cylinders': cylinders,
            'displacement': displacement,
            'horsepower': horsepower,
            'weight': weight,
            'acceleration': acceleration,
            'model_year': model_year,
            'origin_2': 1 if origin == 2 else 0,
            'origin_3': 1 if origin == 3 else 0,
        }

        model = joblib.load("../Regression/ModelsOutcome/mpg_model.pkl")
        scaler = joblib.load("../Regression/ModelsOutcome/mpg_scaler.pkl")
        feature_names = joblib.load("../Regression/ModelsOutcome/mpg_features.pkl")

        input_df = pd.DataFrame([sample])[feature_names]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        self.result_label.config(text=f"Predicted MPG: {round(prediction, 1)}")