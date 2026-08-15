import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from App.scrollable_frame import ScrollableFrame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

YES_NO_MAP = {"No": 0, "Yes": 1}

# مقادیر میانه، برای وقتی کاربر مقداری رو نمی‌دونه
MEDIAN_VALUES = {
    'fractional_shortening': 0.205,
    'epss': 11.0,
    'lvdd': 4.65,
}


class PredictionPageEchocardiogram(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        scroll_container = ScrollableFrame(self)
        scroll_container.pack(fill="both", expand=True)
        content = scroll_container.scrollable_frame

        tk.Label(content, text="Heart Attack Survival Prediction (Echocardiogram)", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts whether a patient will survive at least one year after a heart "
            "attack, based on echocardiogram (heart ultrasound) measurements.\n"
            "Model: Random Forest Classifier."
        )
        tk.Label(content, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(content, text="Model Performance (F1-Score): 47.1%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(content)
        form.pack(pady=10)

        tk.Label(form, text="Age at Heart Attack:").grid(row=0, column=0, sticky="w", pady=5)
        self.age_entry = tk.Entry(form)
        self.age_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Fluid Around the Heart (Pericardial Effusion):").grid(row=1, column=0, sticky="w", pady=5)
        self.effusion_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.effusion_var, values=list(YES_NO_MAP.keys()), state="readonly").grid(row=1, column=1, pady=5)

        tk.Label(form, text="Fractional Shortening (heart pumping efficiency, 0.01-0.61):").grid(row=2, column=0, sticky="w", pady=5)
        self.fs_entry = tk.Entry(form)
        self.fs_entry.grid(row=2, column=1, pady=5)
        self.fs_unknown_var = tk.BooleanVar()
        tk.Checkbutton(form, text="I don't know this value", variable=self.fs_unknown_var).grid(row=2, column=2, padx=5)

        tk.Label(form, text="Mitral Valve Separation Distance / EPSS (0-40):").grid(row=3, column=0, sticky="w", pady=5)
        self.epss_entry = tk.Entry(form)
        self.epss_entry.grid(row=3, column=1, pady=5)
        self.epss_unknown_var = tk.BooleanVar()
        tk.Checkbutton(form, text="I don't know this value", variable=self.epss_unknown_var).grid(row=3, column=2, padx=5)

        tk.Label(form, text="Left Ventricle Chamber Size / LVDD (2.3-6.8):").grid(row=4, column=0, sticky="w", pady=5)
        self.lvdd_entry = tk.Entry(form)
        self.lvdd_entry.grid(row=4, column=1, pady=5)
        self.lvdd_unknown_var = tk.BooleanVar()
        tk.Checkbutton(form, text="I don't know this value", variable=self.lvdd_unknown_var).grid(row=4, column=2, padx=5)

        tk.Label(form, text="Heart Wall Motion Index (1-3):").grid(row=5, column=0, sticky="w", pady=5)
        self.wmi_entry = tk.Entry(form)
        self.wmi_entry.grid(row=5, column=1, pady=5)

        tk.Button(content, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(content, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            age = float(self.age_entry.get())
            effusion = YES_NO_MAP[self.effusion_var.get()]
            wmi = float(self.wmi_entry.get())

            fs_missing = self.fs_unknown_var.get()
            fs = MEDIAN_VALUES['fractional_shortening'] if fs_missing else float(self.fs_entry.get())

            epss_missing = self.epss_unknown_var.get()
            epss = MEDIAN_VALUES['epss'] if epss_missing else float(self.epss_entry.get())

            lvdd_missing = self.lvdd_unknown_var.get()
            lvdd = MEDIAN_VALUES['lvdd'] if lvdd_missing else float(self.lvdd_entry.get())
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'age_at_heart_attack': age,
            'pericardial_effusion': effusion,
            'fractional_shortening': fs,
            'epss': epss,
            'lvdd': lvdd,
            'wall_motion_index': wmi,
            'fractional_shortening_was_missing': 1 if fs_missing else 0,
            'epss_was_missing': 1 if epss_missing else 0,
            'lvdd_was_missing': 1 if lvdd_missing else 0,
        }

        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "heart_attack_rf_model.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "heart_attack_feature.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        result_text = "Likely to Survive (1+ year)" if prediction == 1 else "At Risk (may not survive 1 year)"
        self.result_label.config(text=f"Prediction: {result_text}")