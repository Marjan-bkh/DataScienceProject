import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np


class PredictionPageConcrete(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="regression")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Concrete Compressive Strength Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts the compressive strength of concrete based on its mix components "
            "and curing age.\nModel: Gradient Boosting Regressor."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (R²): 92.4%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Cement (kg/m³):").grid(row=0, column=0, sticky="w", pady=5)
        self.cement_entry = tk.Entry(form)
        self.cement_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Slag (kg/m³):").grid(row=1, column=0, sticky="w", pady=5)
        self.slag_entry = tk.Entry(form)
        self.slag_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="Fly Ash (kg/m³):").grid(row=2, column=0, sticky="w", pady=5)
        self.flyash_entry = tk.Entry(form)
        self.flyash_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Water (kg/m³):").grid(row=3, column=0, sticky="w", pady=5)
        self.water_entry = tk.Entry(form)
        self.water_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Superplasticizer (kg/m³):").grid(row=4, column=0, sticky="w", pady=5)
        self.superplasticizer_entry = tk.Entry(form)
        self.superplasticizer_entry.grid(row=4, column=1, pady=5)

        tk.Label(form, text="Coarse Aggregate (kg/m³):").grid(row=5, column=0, sticky="w", pady=5)
        self.coarse_agg_entry = tk.Entry(form)
        self.coarse_agg_entry.grid(row=5, column=1, pady=5)

        tk.Label(form, text="Fine Aggregate (kg/m³):").grid(row=6, column=0, sticky="w", pady=5)
        self.fine_agg_entry = tk.Entry(form)
        self.fine_agg_entry.grid(row=6, column=1, pady=5)

        tk.Label(form, text="Age (days):").grid(row=7, column=0, sticky="w", pady=5)
        self.age_entry = tk.Entry(form)
        self.age_entry.grid(row=7, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            cement = float(self.cement_entry.get())
            slag = float(self.slag_entry.get())
            flyash = float(self.flyash_entry.get())
            water = float(self.water_entry.get())
            superplasticizer = float(self.superplasticizer_entry.get())
            coarse_agg = float(self.coarse_agg_entry.get())
            fine_agg = float(self.fine_agg_entry.get())
            age = float(self.age_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'cement': cement,
            'slag': slag,
            'flyash': flyash,
            'water': water,
            'superplasticizer': superplasticizer,
            'coarse_agg': coarse_agg,
            'fine_agg': fine_agg,
            'water_cement_ratio': water / cement,
            'total_cementitious': cement + slag + flyash,
            'agg_ratio': fine_agg / coarse_agg,
            'log_age': np.log1p(age),
        }

        model = joblib.load("../../Regression/ModelsOutcome/concrete_model.pkl")
        feature_names = joblib.load("../../Regression/ModelsOutcome/concrete_features.pkl")

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        self.result_label.config(text=f"Predicted Compressive Strength: {round(prediction, 2)} MPa")