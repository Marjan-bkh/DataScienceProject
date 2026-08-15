import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from scrollable_frame import ScrollableFrame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEX_MAP = {"Male": 1, "Female": 2}
YES_NO_MAP = {"No": 1, "Yes": 2}

BINARY_FEATURES = [
    ('STEROID', 'Steroid Treatment'),
    ('ANTIVIRALS', 'Antiviral Treatment'),
    ('FATIGUE', 'Fatigue'),
    ('MALAISE', 'Malaise'),
    ('ANOREXIA', 'Anorexia'),
    ('LIVER_BIG', 'Enlarged Liver'),
    ('LIVER_FIRM', 'Firm Liver'),
    ('SPLEEN_PALPABLE', 'Palpable Spleen'),
    ('SPIDERS', 'Spider Angiomata (skin)'),
    ('ASCITES', 'Ascites (fluid in abdomen)'),
    ('VARICES', 'Varices'),
    ('HISTOLOGY', 'Histology Confirmed'),
]

MEDIAN_VALUES = {
    'ALK_PHOSPHATE': 85.0,
    'PROTIME': 61.0,
}


class PredictionPageHepatitis(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        scroll_container = ScrollableFrame(self)
        scroll_container.pack(fill="both", expand=True)
        content = scroll_container.scrollable_frame

        tk.Label(content, text="Hepatitis Survival Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts whether a hepatitis patient will survive based on clinical and "
            "laboratory findings.\nModel: Random Forest Classifier."
        )
        tk.Label(content, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(content, text="Model Performance (F1-Score): 57.1%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(content)
        form.pack(pady=10)

        row = 0
        tk.Label(form, text="Age:").grid(row=row, column=0, sticky="w", pady=4)
        self.age_entry = tk.Entry(form)
        self.age_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Sex:").grid(row=row, column=0, sticky="w", pady=4)
        self.sex_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.sex_var, values=list(SEX_MAP.keys()), state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        self.binary_vars = {}
        for key, label in BINARY_FEATURES:
            tk.Label(form, text=f"{label}:").grid(row=row, column=0, sticky="w", pady=4)
            var = tk.StringVar()
            ttk.Combobox(form, textvariable=var, values=list(YES_NO_MAP.keys()), state="readonly").grid(row=row, column=1, pady=4)
            self.binary_vars[key] = var
            row += 1

        tk.Label(form, text="Bilirubin (mg/dL, 0.3-8.0):").grid(row=row, column=0, sticky="w", pady=4)
        self.bilirubin_entry = tk.Entry(form)
        self.bilirubin_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Alkaline Phosphatase (26-295):").grid(row=row, column=0, sticky="w", pady=4)
        self.alk_entry = tk.Entry(form)
        self.alk_entry.grid(row=row, column=1, pady=4)
        self.alk_unknown_var = tk.BooleanVar()
        tk.Checkbutton(form, text="I don't know this value", variable=self.alk_unknown_var).grid(row=row, column=2, padx=5)
        row += 1

        tk.Label(form, text="SGOT (liver enzyme, 14-648):").grid(row=row, column=0, sticky="w", pady=4)
        self.sgot_entry = tk.Entry(form)
        self.sgot_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Albumin (2.1-6.4):").grid(row=row, column=0, sticky="w", pady=4)
        self.albumin_entry = tk.Entry(form)
        self.albumin_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Prothrombin Time / PROTIME (0-100):").grid(row=row, column=0, sticky="w", pady=4)
        self.protime_entry = tk.Entry(form)
        self.protime_entry.grid(row=row, column=1, pady=4)
        self.protime_unknown_var = tk.BooleanVar()
        tk.Checkbutton(form, text="I don't know this value", variable=self.protime_unknown_var).grid(row=row, column=2, padx=5)
        row += 1

        tk.Button(content, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(content, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            age = float(self.age_entry.get())
            sex = SEX_MAP[self.sex_var.get()]

            binary_values = {}
            for key, var in self.binary_vars.items():
                binary_values[key] = YES_NO_MAP[var.get()]

            bilirubin = float(self.bilirubin_entry.get())
            sgot = float(self.sgot_entry.get())
            albumin = float(self.albumin_entry.get())

            alk_missing = self.alk_unknown_var.get()
            alk = MEDIAN_VALUES['ALK_PHOSPHATE'] if alk_missing else float(self.alk_entry.get())

            protime_missing = self.protime_unknown_var.get()
            protime = MEDIAN_VALUES['PROTIME'] if protime_missing else float(self.protime_entry.get())
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'AGE': age,
            'SEX': sex,
            'BILIRUBIN': bilirubin,
            'ALK_PHOSPHATE': alk,
            'SGOT': sgot,
            'ALBUMIN': albumin,
            'PROTIME': protime,
            'PROTIME_was_missing': 1 if protime_missing else 0,
            'ALK_PHOSPHATE_was_missing': 1 if alk_missing else 0,
        }
        sample.update(binary_values)

        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "hepatitis_rf_model.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "hepatitis_feature.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        result_text = "Likely to Survive" if prediction == 2 else "High Risk (may not survive)"
        self.result_label.config(text=f"Prediction: {result_text}")