import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class PredictionPageBanknote(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Banknote Authentication", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Determines whether a banknote is genuine or forged based on statistical "
            "features extracted from its wavelet-transformed image.\nModel: Support Vector Machine (SVM)."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (F1-Score): 100%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)


        tk.Label(form, text="Variance of Wavelet Image (-8 to 7):").grid(row=0, column=0, sticky="w", pady=5)
        self.variance_entry = tk.Entry(form)
        self.variance_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Skewness of Wavelet Image (-14 to 13):").grid(row=1, column=0, sticky="w", pady=5)
        self.skewness_entry = tk.Entry(form)
        self.skewness_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="Curtosis of Wavelet Image (-6 to 18):").grid(row=2, column=0, sticky="w", pady=5)
        self.curtosis_entry = tk.Entry(form)
        self.curtosis_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Entropy of Image (-9 to 3):").grid(row=3, column=0, sticky="w", pady=5)
        self.entropy_entry = tk.Entry(form)
        self.entropy_entry.grid(row=3, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def _check_range(self, name, value, min_val, max_val):
        if value < min_val or value > max_val:
            return messagebox.askyesno(
                "Value Out of Range",
                f"{name} = {value} is outside the typical range ({min_val} to {max_val}).\n"
                f"The prediction may be less reliable. Continue anyway?"
            )
        return True

    def on_predict(self):
        try:
            variance = float(self.variance_entry.get())
            skewness = float(self.skewness_entry.get())
            curtosis = float(self.curtosis_entry.get())
            entropy = float(self.entropy_entry.get())
            if not self._check_range("Variance", variance, -8, 7):
                return
            if not self._check_range("Skewness", skewness, -14, 13):
                return
            if not self._check_range("Curtosis", curtosis, -6, 18):
                return
            if not self._check_range("Entropy", entropy, -9, 3):
                return
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'variance': variance,
            'skewness': skewness,
            'curtosis': curtosis,
            'entropy': entropy,
        }

        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "banknote_svm_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "banknote_scaler.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "banknote_feature_columns.pkl"))


        input_df = pd.DataFrame([sample])[feature_names]
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        prediction = model.predict(input_scaled_df)[0]

        result_text = "Genuine" if prediction == 0 else "Forged"
        self.result_label.config(text=f"Prediction: {result_text} Banknote")