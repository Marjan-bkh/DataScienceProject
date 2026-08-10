import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RISK_OPTIONS = ["Negative (N)", "Average (A)", "Positive (P)"]
RISK_MAP = {"Negative (N)": "N", "Average (A)": "A", "Positive (P)": "P"}

FEATURE_LABELS = {
    'IR': "Industrial Risk",
    'MR': "Management Risk",
    'FF': "Financial Flexibility",
    'CR': "Credibility",
    'CO': "Competitiveness",
    'OP': "Operating Risk",
}


class PredictionPageQualitativeBankruptcy(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Qualitative Bankruptcy Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts whether a company is at risk of bankruptcy based on qualitative "
            "expert assessments of key risk factors.\nModel: Random Forest Classifier."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (Accuracy): 100%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        self.vars = {}
        for i, (key, label) in enumerate(FEATURE_LABELS.items()):
            tk.Label(form, text=f"{label}:").grid(row=i, column=0, sticky="w", pady=5)
            var = tk.StringVar()
            ttk.Combobox(form, textvariable=var, values=RISK_OPTIONS, state="readonly").grid(row=i, column=1, pady=5)
            self.vars[key] = var

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            sample = {}
            for key, var in self.vars.items():
                selected = var.get()
                if selected == "":
                    raise ValueError
                sample[key] = RISK_MAP[selected]
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        risk_mapping = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "bankruptcy_risk_mapping.pkl"))
        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "bankruptcy_model.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "bankruptcy_feature.pkl"))

        encoded_sample = {key: risk_mapping[value] for key, value in sample.items()}
        input_df = pd.DataFrame([encoded_sample])[feature_names]

        prediction = model.predict(input_df)[0]

        result_text = "Bankruptcy Risk" if prediction == "B" else "Non-Bankruptcy"
        self.result_label.config(text=f"Prediction: {result_text}")