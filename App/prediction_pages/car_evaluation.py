import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BUYING_OPTIONS = ['low', 'med', 'high', 'vhigh']
MAINT_OPTIONS = ['low', 'med', 'high', 'vhigh']
DOORS_OPTIONS = ['2', '3', '4', '5more']
PERSONS_OPTIONS = ['2', '4', 'more']
LUG_BOOT_OPTIONS = ['small', 'med', 'big']
SAFETY_OPTIONS = ['low', 'med', 'high']


class PredictionPageCarEvaluation(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Car Evaluation", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Evaluates the overall acceptability of a car based on its price, safety, "
            "and comfort attributes.\nModel: Gradient Boosting Classifier."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (F1-macro): 98.97%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Buying Price:").grid(row=0, column=0, sticky="w", pady=5)
        self.buying_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.buying_var, values=BUYING_OPTIONS, state="readonly").grid(row=0, column=1, pady=5)

        tk.Label(form, text="Maintenance Price:").grid(row=1, column=0, sticky="w", pady=5)
        self.maint_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.maint_var, values=MAINT_OPTIONS, state="readonly").grid(row=1, column=1, pady=5)

        tk.Label(form, text="Number of Doors:").grid(row=2, column=0, sticky="w", pady=5)
        self.doors_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.doors_var, values=DOORS_OPTIONS, state="readonly").grid(row=2, column=1, pady=5)

        tk.Label(form, text="Passenger Capacity:").grid(row=3, column=0, sticky="w", pady=5)
        self.persons_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.persons_var, values=PERSONS_OPTIONS, state="readonly").grid(row=3, column=1, pady=5)

        tk.Label(form, text="Luggage Boot Size:").grid(row=4, column=0, sticky="w", pady=5)
        self.lug_boot_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.lug_boot_var, values=LUG_BOOT_OPTIONS, state="readonly").grid(row=4, column=1, pady=5)

        tk.Label(form, text="Safety Rating:").grid(row=5, column=0, sticky="w", pady=5)
        self.safety_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.safety_var, values=SAFETY_OPTIONS, state="readonly").grid(row=5, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            sample = {
                'buying': self.buying_var.get(),
                'maint': self.maint_var.get(),
                'doors': self.doors_var.get(),
                'persons': self.persons_var.get(),
                'lug_boot': self.lug_boot_var.get(),
                'safety': self.safety_var.get(),
            }
            if "" in sample.values():
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "car_evaluation_gb_model.pkl"))
        encoder = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "car_evaluation_encoder.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "car_evaluation_feature.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        input_encoded = encoder.transform(input_df)
        input_encoded_df = pd.DataFrame(input_encoded, columns=feature_names)

        prediction = model.predict(input_encoded_df)[0]

        self.result_label.config(text=f"Predicted Car Acceptability: {prediction}")