import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

WEEK_MAP = {"1st week": 1, "2nd week": 2, "3rd week": 3, "4th week": 4, "5th week": 5}
DAY_MAP = {"Monday": 2, "Tuesday": 3, "Wednesday": 4, "Thursday": 5, "Friday": 6}


class PredictionPageDailyOrders(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="regression")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Daily Demand Forecasting Orders", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts the total number of daily orders for a logistics company based on "
            "sector-specific order volumes and calendar information.\nModel: Lasso Regression."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (R²): 72.7%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Week of the Month:").grid(row=0, column=0, sticky="w", pady=5)
        self.week_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.week_var, values=list(WEEK_MAP.keys()), state="readonly").grid(row=0, column=1, pady=5)

        tk.Label(form, text="Day of the Week:").grid(row=1, column=0, sticky="w", pady=5)
        self.day_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.day_var, values=list(DAY_MAP.keys()), state="readonly").grid(row=1, column=1, pady=5)

        tk.Label(form, text="Fiscal Sector Orders:").grid(row=2, column=0, sticky="w", pady=5)
        self.fiscal_entry = tk.Entry(form)
        self.fiscal_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Traffic Controller Sector Orders:").grid(row=3, column=0, sticky="w", pady=5)
        self.traffic_entry = tk.Entry(form)
        self.traffic_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Banking Orders (1):").grid(row=4, column=0, sticky="w", pady=5)
        self.bank1_entry = tk.Entry(form)
        self.bank1_entry.grid(row=4, column=1, pady=5)

        tk.Label(form, text="Banking Orders (2):").grid(row=5, column=0, sticky="w", pady=5)
        self.bank2_entry = tk.Entry(form)
        self.bank2_entry.grid(row=5, column=1, pady=5)

        tk.Label(form, text="Banking Orders (3):").grid(row=6, column=0, sticky="w", pady=5)
        self.bank3_entry = tk.Entry(form)
        self.bank3_entry.grid(row=6, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            week = WEEK_MAP[self.week_var.get()]
            day = DAY_MAP[self.day_var.get()]
            fiscal = float(self.fiscal_entry.get())
            traffic = float(self.traffic_entry.get())
            bank1 = float(self.bank1_entry.get())
            bank2 = float(self.bank2_entry.get())
            bank3 = float(self.bank3_entry.get())
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'Week of the month (first week, second, third, fourth or fifth week': week,
            'Day of the week (Monday to Friday)': day,
            'Fiscal sector orders': fiscal,
            'Orders from the traffic controller sector': traffic,
            'Banking orders (1)': bank1,
            'Banking orders (2)': bank2,
            'Banking orders (3)': bank3,
        }

        model = joblib.load("../../Regression/ModelsOutcome/daily_orders_model.pkl")
        scaler = joblib.load("../../Regression/ModelsOutcome/daily_orders_scaler.pkl")
        feature_names = joblib.load("../../Regression/ModelsOutcome/daily_orders_features.pkl")

        input_df = pd.DataFrame([sample])[feature_names]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        self.result_label.config(text=f"Predicted Total Orders: {round(prediction, 1)}")