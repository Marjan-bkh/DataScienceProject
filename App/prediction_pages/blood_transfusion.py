import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import  os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PredictionPageBloodTransfusion(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Blood Transfusion Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts whether a blood donor will donate again based on their donation "
            "history.\nModel: Random Forest Classifier."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (F1-Score): 53.7%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Recency (months since last donation, 0-74):").grid(row=0, column=0, sticky="w", pady=5)
        self.recency_entry = tk.Entry(form)
        self.recency_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Frequency (total number of donations, 1-50):").grid(row=1, column=0, sticky="w", pady=5)
        self.frequency_entry = tk.Entry(form)
        self.frequency_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="Time (months since first donation, 2-98):").grid(row=2, column=0, sticky="w", pady=5)
        self.time_entry = tk.Entry(form)
        self.time_entry.grid(row=2, column=1, pady=5)

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
            recency = float(self.recency_entry.get())
            frequency = float(self.frequency_entry.get())
            time = float(self.time_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        if not self._check_range("Recency", recency, 0, 74):
            return
        if not self._check_range("Frequency", frequency, 1, 50):
            return
        if not self._check_range("Time", time, 2, 98):
            return

        sample = {
            'Recency': recency,
            'Frequency': frequency,
            'Time': time,
            'Frequency_per_Time': frequency / time,
        }

        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "blood_transfusion_model.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "blood_transfusion_feature_columns.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        result_text = "Will Donate Again" if prediction == 1 else "Will Not Donate Again"
        self.result_label.config(text=f"Prediction: {result_text}")