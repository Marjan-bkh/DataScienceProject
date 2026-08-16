import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PredictionPageCovidRisk(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="COVID-19 Weekly Risk Level Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts a country's COVID-19 risk level (Low/Medium/High) for the upcoming "
            "week based on recent case growth and fatality trends.\n"
            "Model: Gradient Boosting Classifier. Predicting epidemic trends is inherently "
            "difficult; accuracy reflects real-world uncertainty."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (F1-macro): 58.5%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Case Growth Rate — 1 week ago (%):").grid(row=0, column=0, sticky="w", pady=5)
        self.growth1_entry = tk.Entry(form)
        self.growth1_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Case Growth Rate — 2 weeks ago (%):").grid(row=1, column=0, sticky="w", pady=5)
        self.growth2_entry = tk.Entry(form)
        self.growth2_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="Case Growth Rate — 3 weeks ago (%):").grid(row=2, column=0, sticky="w", pady=5)
        self.growth3_entry = tk.Entry(form)
        self.growth3_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Case Fatality Rate — 1 week ago (%):").grid(row=3, column=0, sticky="w", pady=5)
        self.cfr1_entry = tk.Entry(form)
        self.cfr1_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Case Fatality Rate — 2 weeks ago (%):").grid(row=4, column=0, sticky="w", pady=5)
        self.cfr2_entry = tk.Entry(form)
        self.cfr2_entry.grid(row=4, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            growth1 = float(self.growth1_entry.get()) / 100
            growth2 = float(self.growth2_entry.get()) / 100
            growth3 = float(self.growth3_entry.get()) / 100
            cfr1 = float(self.cfr1_entry.get()) / 100
            cfr2 = float(self.cfr2_entry.get()) / 100
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        # Growth_Rate_rolling3 میانگین نرخ رشد سه هفته‌ی قبل (۱، ۲ و ۳ هفته‌ی پیش) است
        growth_rolling3 = (growth1 + growth2 + growth3) / 3

        sample = {
            'Growth_Rate_lag1': growth1,
            'Growth_Rate_lag2': growth2,
            'CFR_lag1': cfr1,
            'CFR_lag2': cfr2,
            'Growth_Rate_rolling3': growth_rolling3,
        }

        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "covid_risk_gb_model.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "covid_risk_features.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        self.result_label.config(text=f"Predicted Risk Level: {prediction}")