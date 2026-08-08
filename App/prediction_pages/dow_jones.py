import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd


class PredictionPageDowJones(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="regression")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Dow Jones Index — Next Week's Price Change", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts the percent change in stock price for next week based on this week's "
            "trading data.\nModel: Random Forest Regressor (tuned)."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (R²): -0.51% — stock prices are inherently hard to predict; "
                             "this near-zero/negative R² reflects real market unpredictability, not a modeling error.",
                 font=("Arial", 9, "italic"), fg="gray8", wraplength=550, justify="center").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Quarter (1 or 2):").grid(row=0, column=0, sticky="w", pady=5)
        self.quarter_entry = tk.Entry(form)
        self.quarter_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Open Price ($):").grid(row=1, column=0, sticky="w", pady=5)
        self.open_entry = tk.Entry(form)
        self.open_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="High Price ($):").grid(row=2, column=0, sticky="w", pady=5)
        self.high_entry = tk.Entry(form)
        self.high_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Low Price ($):").grid(row=3, column=0, sticky="w", pady=5)
        self.low_entry = tk.Entry(form)
        self.low_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Close Price ($):").grid(row=4, column=0, sticky="w", pady=5)
        self.close_entry = tk.Entry(form)
        self.close_entry.grid(row=4, column=1, pady=5)

        tk.Label(form, text="Volume:").grid(row=5, column=0, sticky="w", pady=5)
        self.volume_entry = tk.Entry(form)
        self.volume_entry.grid(row=5, column=1, pady=5)

        tk.Label(form, text="Previous Week's Volume:").grid(row=6, column=0, sticky="w", pady=5)
        self.prev_volume_entry = tk.Entry(form)
        self.prev_volume_entry.grid(row=6, column=1, pady=5)

        tk.Label(form, text="% Change in Price (this week):").grid(row=7, column=0, sticky="w", pady=5)
        self.pct_change_price_entry = tk.Entry(form)
        self.pct_change_price_entry.grid(row=7, column=1, pady=5)

        tk.Label(form, text="% Change in Price (last week):").grid(row=8, column=0, sticky="w", pady=5)
        self.prev_pct_change_price_entry = tk.Entry(form)
        self.prev_pct_change_price_entry.grid(row=8, column=1, pady=5)

        tk.Label(form, text="% Change in Volume vs Last Week:").grid(row=9, column=0, sticky="w", pady=5)
        self.pct_change_volume_entry = tk.Entry(form)
        self.pct_change_volume_entry.grid(row=9, column=1, pady=5)

        tk.Label(form, text="Days to Next Dividend:").grid(row=10, column=0, sticky="w", pady=5)
        self.days_dividend_entry = tk.Entry(form)
        self.days_dividend_entry.grid(row=10, column=1, pady=5)

        tk.Label(form, text="% Return on Next Dividend:").grid(row=11, column=0, sticky="w", pady=5)
        self.pct_return_dividend_entry = tk.Entry(form)
        self.pct_return_dividend_entry.grid(row=11, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            quarter = int(self.quarter_entry.get())
            open_price = float(self.open_entry.get())
            high = float(self.high_entry.get())
            low = float(self.low_entry.get())
            close = float(self.close_entry.get())
            volume = float(self.volume_entry.get())
            prev_volume = float(self.prev_volume_entry.get())
            pct_change_price = float(self.pct_change_price_entry.get())
            prev_pct_change_price = float(self.prev_pct_change_price_entry.get())
            pct_change_volume = float(self.pct_change_volume_entry.get())
            days_dividend = float(self.days_dividend_entry.get())
            pct_return_dividend = float(self.pct_return_dividend_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'quarter': quarter,
            'volume': volume,
            'percent_change_price': pct_change_price,
            'percent_change_volume_over_last_wk': pct_change_volume,
            'previous_weeks_volume': prev_volume,
            'days_to_next_dividend': days_dividend,
            'percent_return_next_dividend': pct_return_dividend,
            'price_range': (high - low) / open_price,
            'open_close_change': (close - open_price) / open_price,
            'prev_percent_change_price': prev_pct_change_price,
            'volume_change_ratio': volume / prev_volume,
        }

        model = joblib.load("../../Regression/ModelsOutcome/dow_jones_rf_model.pkl")
        feature_names = joblib.load("../../Regression/ModelsOutcome/dow_jones_feature_columns.pkl")

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        self.result_label.config(text=f"Predicted Next Week's Price Change: {round(prediction, 2)}%")