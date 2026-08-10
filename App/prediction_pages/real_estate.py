import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}


class PredictionPageRealEstate(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="regression")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Real Estate Price Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts the price per unit area of a property in Taiwan based on its age, "
            "location, and nearby amenities.\nModel: Linear Regression."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (R²): 56.0%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Transaction Year:").grid(row=0, column=0, sticky="w", pady=5)
        self.year_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.year_var, values=["2012", "2013"], state="readonly").grid(row=0, column=1, pady=5)

        tk.Label(form, text="Transaction Month:").grid(row=1, column=0, sticky="w", pady=5)
        self.month_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.month_var, values=list(MONTH_MAP.keys()), state="readonly").grid(row=1, column=1, pady=5)

        tk.Label(form, text="House Age (years):").grid(row=2, column=0, sticky="w", pady=5)
        self.house_age_entry = tk.Entry(form)
        self.house_age_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Distance to Nearest MRT (meters):").grid(row=3, column=0, sticky="w", pady=5)
        self.distance_entry = tk.Entry(form)
        self.distance_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Number of Convenience Stores Nearby:").grid(row=4, column=0, sticky="w", pady=5)
        self.stores_entry = tk.Entry(form)
        self.stores_entry.grid(row=4, column=1, pady=5)

        tk.Label(form, text="Latitude:").grid(row=5, column=0, sticky="w", pady=5)
        self.latitude_entry = tk.Entry(form)
        self.latitude_entry.grid(row=5, column=1, pady=5)

        tk.Label(form, text="Longitude:").grid(row=6, column=0, sticky="w", pady=5)
        self.longitude_entry = tk.Entry(form)
        self.longitude_entry.grid(row=6, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            year = int(self.year_var.get())
            month = MONTH_MAP[self.month_var.get()]
            house_age = float(self.house_age_entry.get())
            distance = float(self.distance_entry.get())
            stores = int(self.stores_entry.get())
            latitude = float(self.latitude_entry.get())
            longitude = float(self.longitude_entry.get())
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        transaction_date = year + (month - 1) / 12

        sample = {
            'TransactionDate': transaction_date,
            'HouseAge': house_age,
            'DistanceToMRT': distance,
            'ConvenienceStores': stores,
            'Latitude': latitude,
            'Longitude': longitude,
        }

        model = joblib.load("../../Regression/ModelsOutcome/real_estate_model.pkl")
        feature_names = joblib.load("../../Regression/ModelsOutcome/real_estate_feature_columns.pkl")

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        self.result_label.config(text=f"Predicted Price: {round(prediction, 2)} (per unit area, ~3.3 m²)")