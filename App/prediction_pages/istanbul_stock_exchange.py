import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PredictionIstanbulStockPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="regression")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Istanbul Stock Exchange Return Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts the daily return of the Istanbul Stock Exchange (ISE) using the "
            "daily returns of major international indices (S&P 500, DAX, FTSE 100, "
            "Nikkei 225, Bovespa, MSCI Emerging Markets). \n Model: Linear Regression."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(
            pady=(0, 5))

        tk.Label(self, text="Model Accuracy (Cross-Validated R²): 42%", font=("Arial", 10, "italic"), fg="gray8").pack(
            pady=(0, 5))

        hint_text = "Enter daily returns as decimals (e.g. 0.02 = +2%, -0.015 = -1.5%). Typical range: -0.06 to 0.07."
        tk.Label(self, text=hint_text, font=("Arial", 9), fg="gray40", wraplength=550, justify="center").pack(
            pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        feature_labels = {
            'SP': 'S&P 500 Daily Return',
            'DAX': 'DAX Daily Return',
            'FTSE': 'FTSE 100 Daily Return',
            'NIKKEI': 'Nikkei 225 Daily Return',
            'BOVESPA': 'Bovespa Daily Return',
            'EM': 'MSCI Emerging Markets Daily Return'
        }

        self.entries = {}
        for i, (col, label_text) in enumerate(feature_labels.items()):
            tk.Label(form, text=label_text + ":").grid(row=i, column=0, sticky="w", pady=5)
            entry = tk.Entry(form)
            entry.grid(row=i, column=1, pady=5)
            self.entries[col] = entry

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self, dataset=None):
        pass

    def on_predict(self):
        try:
            values = {col: float(entry.get()) for col, entry in self.entries.items()}
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        saved = joblib.load(os.path.join(BASE_DIR, "Regression", "ModelsOutcome", "istanbul_stock_linear_regression_model.pkl"))
        model = saved['model']
        feature_names = saved['feature_names']

        input_df = pd.DataFrame([values])[feature_names]
        prediction = model.predict(input_df)[0]

        self.result_label.config(text=f"Predicted ISE Daily Return: {prediction:.5f}")


from App.category_page import CategoryPage