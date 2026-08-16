import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PredictionPageWholesaleCustomers(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="clustering")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Wholesale Customer Segmentation", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Segments wholesale customers into business types based on their annual "
            "spending across product categories.\nModel: K-Means Clustering (k=2)."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Cluster 0 aligns 96.8% with Hotel/Restaurant/Café businesses; "
                             "Cluster 1 aligns mostly with Retail businesses.",
                 font=("Arial", 9, "italic"), fg="gray8", wraplength=550, justify="center").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Annual Spending — Fresh Products ($):").grid(row=0, column=0, sticky="w", pady=5)
        self.fresh_entry = tk.Entry(form)
        self.fresh_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Annual Spending — Milk ($):").grid(row=1, column=0, sticky="w", pady=5)
        self.milk_entry = tk.Entry(form)
        self.milk_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="Annual Spending — Grocery ($):").grid(row=2, column=0, sticky="w", pady=5)
        self.grocery_entry = tk.Entry(form)
        self.grocery_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Annual Spending — Frozen Products ($):").grid(row=3, column=0, sticky="w", pady=5)
        self.frozen_entry = tk.Entry(form)
        self.frozen_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Annual Spending — Detergents/Paper ($):").grid(row=4, column=0, sticky="w", pady=5)
        self.detergents_entry = tk.Entry(form)
        self.detergents_entry.grid(row=4, column=1, pady=5)

        tk.Label(form, text="Annual Spending — Delicatessen ($):").grid(row=5, column=0, sticky="w", pady=5)
        self.deli_entry = tk.Entry(form)
        self.deli_entry.grid(row=5, column=1, pady=5)

        tk.Button(self, text="Find Customer Segment", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black", wraplength=550, justify="center")
        self.result_label.pack(pady=10, padx=20)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            fresh = float(self.fresh_entry.get())
            milk = float(self.milk_entry.get())
            grocery = float(self.grocery_entry.get())
            frozen = float(self.frozen_entry.get())
            detergents = float(self.detergents_entry.get())
            deli = float(self.deli_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'Fresh': np.log1p(fresh),
            'Milk': np.log1p(milk),
            'Grocery': np.log1p(grocery),
            'Frozen': np.log1p(frozen),
            'Detergents_Paper': np.log1p(detergents),
            'Delicassen': np.log1p(deli),
        }

        model = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "wholesale_kmeans_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "wholesale_scaler.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "wholesale_feature_columns.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

        cluster = model.predict(input_scaled_df)[0]

        if cluster == 0:
            result_text = "Segment: Hotel/Restaurant/Café (Horeca) — high fresh food spending relative to grocery/detergent spending."
        else:
            result_text = "Segment: Retail — higher spending on milk, grocery, and detergent/paper products."
        self.result_label.config(text=result_text)