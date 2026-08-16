import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PredictionPageRoomOccupancy(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Room Occupancy Detection", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Detects whether a room is occupied based on ambient sensor readings.\n"
            "Model: Logistic Regression."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (F1-Score): 96.2%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Temperature (°C):").grid(row=0, column=0, sticky="w", pady=5)
        self.temp_entry = tk.Entry(form)
        self.temp_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Light (Lux):").grid(row=1, column=0, sticky="w", pady=5)
        self.light_entry = tk.Entry(form)
        self.light_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="CO2 (ppm):").grid(row=2, column=0, sticky="w", pady=5)
        self.co2_entry = tk.Entry(form)
        self.co2_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Humidity Ratio (kg water-vapor/kg air):").grid(row=3, column=0, sticky="w", pady=5)
        self.humidity_ratio_entry = tk.Entry(form)
        self.humidity_ratio_entry.grid(row=3, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            temperature = float(self.temp_entry.get())
            light = float(self.light_entry.get())
            co2 = float(self.co2_entry.get())
            humidity_ratio = float(self.humidity_ratio_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'Temperature': temperature,
            'Light': light,
            'CO2': co2,
            'HumidityRatio': humidity_ratio,
        }

        pipeline = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "room_occupancy_logreg_model.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "room_occupancy_feature_columns.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = pipeline.predict(input_df)[0]

        result_text = "Occupied" if prediction == 1 else "Empty"
        self.result_label.config(text=f"Room Status: {result_text}")