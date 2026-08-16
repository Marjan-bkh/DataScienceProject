import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLUSTER_LABELS = {
    3: "Low-Consumption Day — likely an empty house or minimal activity.",
    1: "Typical Day — dominated by heating/cooling (water heater & AC) usage.",
    0: "Variable Day — high fluctuation in usage, notable kitchen activity.",
    2: "High-Consumption Day — heavy usage overall, notable laundry activity.",
}


class PredictionPagePowerConsumption(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="clustering")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Household Power Consumption Day-Type Clustering", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Groups a household's daily electricity usage into behavioral patterns based "
            "on power draw and appliance usage ratios.\nModel: K-Means Clustering (k=4)."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Enter summary statistics for one full day of household power usage.",
                 font=("Arial", 9, "italic"), fg="gray8", wraplength=550, justify="center").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Average Active Power (kW):").grid(row=0, column=0, sticky="w", pady=5)
        self.mean_power_entry = tk.Entry(form)
        self.mean_power_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Maximum Active Power (kW):").grid(row=1, column=0, sticky="w", pady=5)
        self.max_power_entry = tk.Entry(form)
        self.max_power_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="Average Reactive Power (kW):").grid(row=2, column=0, sticky="w", pady=5)
        self.reactive_power_entry = tk.Entry(form)
        self.reactive_power_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Power Variability (Std/Mean ratio, e.g. 0.9):").grid(row=3, column=0, sticky="w", pady=5)
        self.cv_entry = tk.Entry(form)
        self.cv_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Kitchen Energy (Watt-hours):").grid(row=4, column=0, sticky="w", pady=5)
        self.kitchen_entry = tk.Entry(form)
        self.kitchen_entry.grid(row=4, column=1, pady=5)

        tk.Label(form, text="Laundry Energy (Watt-hours):").grid(row=5, column=0, sticky="w", pady=5)
        self.laundry_entry = tk.Entry(form)
        self.laundry_entry.grid(row=5, column=1, pady=5)

        tk.Label(form, text="Water Heater / AC Energy (Watt-hours):").grid(row=6, column=0, sticky="w", pady=5)
        self.heater_entry = tk.Entry(form)
        self.heater_entry.grid(row=6, column=1, pady=5)

        tk.Button(self, text="Find Day Type", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black", wraplength=550, justify="center")
        self.result_label.pack(pady=10, padx=20)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            mean_power = float(self.mean_power_entry.get())
            max_power = float(self.max_power_entry.get())
            reactive_power = float(self.reactive_power_entry.get())
            cv = float(self.cv_entry.get())
            kitchen = float(self.kitchen_entry.get())
            laundry = float(self.laundry_entry.get())
            heater = float(self.heater_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        total_sub = kitchen + laundry + heater
        if total_sub == 0:
            messagebox.showerror("Input Error", "At least one appliance energy value must be greater than 0.")
            return

        sample = {
            'Global_active_power_mean': mean_power,
            'Global_active_power_max': max_power,
            'Global_reactive_power_mean': reactive_power,
            'Sub_metering_1_ratio': kitchen / total_sub,
            'Sub_metering_2_ratio': laundry / total_sub,
            'Sub_metering_3_ratio': heater / total_sub,
            'Global_active_power_cv': cv,
        }

        model = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "power_consumption_kmeans_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "power_consumption_scaler.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "power_consumption_feature_columns.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

        cluster = model.predict(input_scaled_df)[0]
        label = CLUSTER_LABELS.get(cluster, f"Cluster {cluster}")

        self.result_label.config(text=f"Day Type: {label}")