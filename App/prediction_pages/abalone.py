import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

SEX_MAP = {"Male": 1, "Female": 2, "Infant": 3}


class PredictionPageAbalone(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="regression")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Abalone Age Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts the age of an abalone (in number of rings) based on its physical "
            "measurements.\nModel: Linear Regression."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (R²): 55.9%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Sex:").grid(row=0, column=0, sticky="w", pady=5)
        self.sex_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.sex_var, values=list(SEX_MAP.keys()), state="readonly").grid(row=0, column=1, pady=5)

        tk.Label(form, text="Length (mm):").grid(row=1, column=0, sticky="w", pady=5)
        self.length_entry = tk.Entry(form)
        self.length_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="Diameter (mm):").grid(row=2, column=0, sticky="w", pady=5)
        self.diameter_entry = tk.Entry(form)
        self.diameter_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Height (mm):").grid(row=3, column=0, sticky="w", pady=5)
        self.height_entry = tk.Entry(form)
        self.height_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Whole Weight (g):").grid(row=4, column=0, sticky="w", pady=5)
        self.whole_weight_entry = tk.Entry(form)
        self.whole_weight_entry.grid(row=4, column=1, pady=5)

        tk.Label(form, text="Shucked Weight (g):").grid(row=5, column=0, sticky="w", pady=5)
        self.shucked_weight_entry = tk.Entry(form)
        self.shucked_weight_entry.grid(row=5, column=1, pady=5)

        tk.Label(form, text="Viscera Weight (g):").grid(row=6, column=0, sticky="w", pady=5)
        self.viscera_weight_entry = tk.Entry(form)
        self.viscera_weight_entry.grid(row=6, column=1, pady=5)

        tk.Label(form, text="Shell Weight (g):").grid(row=7, column=0, sticky="w", pady=5)
        self.shell_weight_entry = tk.Entry(form)
        self.shell_weight_entry.grid(row=7, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            sex = SEX_MAP[self.sex_var.get()]
            length = float(self.length_entry.get())
            diameter = float(self.diameter_entry.get())
            height = float(self.height_entry.get())
            whole_weight = float(self.whole_weight_entry.get())
            shucked_weight = float(self.shucked_weight_entry.get())
            viscera_weight = float(self.viscera_weight_entry.get())
            shell_weight = float(self.shell_weight_entry.get())
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'Sex': sex,
            'Length': length,
            'Diameter': diameter,
            'Height': height,
            'WholeWeight': whole_weight,
            'ShuckedWeight': shucked_weight,
            'VisceraWeight': viscera_weight,
            'ShellWeight': shell_weight,
        }

        model = joblib.load("../../Regression/ModelsOutcome/abalone_model.pkl")
        feature_names = joblib.load("../../Regression/ModelsOutcome/abalone_feature_columns.pkl")

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        # حلقه‌ها + ۱.۵ تقریباً معادل سن واقعی (بر اساس مستندات دیتاست)
        self.result_label.config(text=f"Predicted Rings: {round(prediction, 1)} (≈ Age: {round(prediction + 1.5, 1)} years)")