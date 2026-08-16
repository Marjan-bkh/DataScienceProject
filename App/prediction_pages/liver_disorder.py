import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PredictionPageLiverDisorder(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="clustering")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Liver Disorder Profile Clustering", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Groups patients into clusters based on blood test results and alcohol "
            "consumption, revealing patterns associated with liver health.\n"
            "Model: K-Means Clustering (k=2)."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Cluster Separation: Group 1 shows notably higher liver enzyme "
                             "levels and alcohol consumption on average than Group 0.",
                 font=("Arial", 9, "italic"), fg="gray8", wraplength=550, justify="center").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Mean Corpuscular Volume / MCV (65-103):").grid(row=0, column=0, sticky="w", pady=5)
        self.mcv_entry = tk.Entry(form)
        self.mcv_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Alkaline Phosphatase / ALKPHOS (23-138):").grid(row=1, column=0, sticky="w", pady=5)
        self.alkphos_entry = tk.Entry(form)
        self.alkphos_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="SGPT (Liver Enzyme, 4-155):").grid(row=2, column=0, sticky="w", pady=5)
        self.sgpt_entry = tk.Entry(form)
        self.sgpt_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="SGOT (Liver Enzyme, 5-82):").grid(row=3, column=0, sticky="w", pady=5)
        self.sgot_entry = tk.Entry(form)
        self.sgot_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Gamma-GT (Liver Enzyme, 5-297):").grid(row=4, column=0, sticky="w", pady=5)
        self.gammagt_entry = tk.Entry(form)
        self.gammagt_entry.grid(row=4, column=1, pady=5)

        tk.Label(form, text="Alcoholic Drinks (per day, 0-20):").grid(row=5, column=0, sticky="w", pady=5)
        self.drinks_entry = tk.Entry(form)
        self.drinks_entry.grid(row=5, column=1, pady=5)

        tk.Button(self, text="Find My Group", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black", wraplength=550,justify="center")
        self.result_label.pack(pady=10, padx=20)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            mcv = float(self.mcv_entry.get())
            alkphos = float(self.alkphos_entry.get())
            sgpt = float(self.sgpt_entry.get())
            sgot = float(self.sgot_entry.get())
            gammagt = float(self.gammagt_entry.get())
            drinks = float(self.drinks_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'mcv': mcv,
            'alkphos': np.log1p(alkphos),
            'sgpt': np.log1p(sgpt),
            'sgot': np.log1p(sgot),
            'gammagt': np.log1p(gammagt),
            'drinks': np.log1p(drinks),
        }

        model = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "liver_disorder_kmeans_k2_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "liver_disorder_scaler.pkl"))
        feature_names = list(scaler.feature_names_in_)

        input_df = pd.DataFrame([sample])[feature_names]
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        cluster = model.predict(input_scaled_df)[0]

        if cluster == 1:
            result_text = "Group 1 — Profile shows higher liver enzyme levels and alcohol consumption"
        else:
            result_text = "Group 0 — Profile is closer to typical/lower liver enzyme levels"
        self.result_label.config(text=f"Cluster Assignment: {result_text}")