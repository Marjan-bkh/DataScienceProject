import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from App.scrollable_frame import ScrollableFrame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CP_OPTIONS = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
RESTECG_OPTIONS = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
SLOPE_OPTIONS = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
THAL_OPTIONS = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
YES_NO_MAP = {"No": 0, "Yes": 1}
SEX_MAP = {"Male": 1, "Female": 0}


class PredictionPageHeartDisease(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)
        scroll_container = ScrollableFrame(self)
        scroll_container.pack(fill="both", expand=True)
        content = scroll_container.scrollable_frame
        tk.Label(content, text="Heart Disease Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts the presence of heart disease based on clinical measurements and "
            "test results.\nModel: Logistic Regression."
        )
        tk.Label(content, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(content, text="Model Performance (Accuracy): 86.7%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(content)
        form.pack(pady=10)

        tk.Label(form, text="Age:").grid(row=0, column=0, sticky="w", pady=5)
        self.age_entry = tk.Entry(form)
        self.age_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Sex:").grid(row=1, column=0, sticky="w", pady=5)
        self.sex_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.sex_var, values=list(SEX_MAP.keys()), state="readonly").grid(row=1, column=1, pady=5)

        tk.Label(form, text="Resting Blood Pressure (mmHg, 94-200):").grid(row=2, column=0, sticky="w", pady=5)
        self.trestbps_entry = tk.Entry(form)
        self.trestbps_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Cholesterol (mg/dl, 126-564):").grid(row=3, column=0, sticky="w", pady=5)
        self.chol_entry = tk.Entry(form)
        self.chol_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Fasting Blood Sugar > 120 mg/dl:").grid(row=4, column=0, sticky="w", pady=5)
        self.fbs_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.fbs_var, values=list(YES_NO_MAP.keys()), state="readonly").grid(row=4, column=1, pady=5)

        tk.Label(form, text="Max Heart Rate Achieved (71-202):").grid(row=5, column=0, sticky="w", pady=5)
        self.thalach_entry = tk.Entry(form)
        self.thalach_entry.grid(row=5, column=1, pady=5)

        tk.Label(form, text="Exercise-Induced Angina:").grid(row=6, column=0, sticky="w", pady=5)
        self.exang_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.exang_var, values=list(YES_NO_MAP.keys()), state="readonly").grid(row=6, column=1, pady=5)

        tk.Label(form, text="ST Depression (oldpeak, 0-6.2):").grid(row=7, column=0, sticky="w", pady=5)
        self.oldpeak_entry = tk.Entry(form)
        self.oldpeak_entry.grid(row=7, column=1, pady=5)

        tk.Label(form, text="Number of Major Vessels (0-3):").grid(row=8, column=0, sticky="w", pady=5)
        self.ca_entry = tk.Entry(form)
        self.ca_entry.grid(row=8, column=1, pady=5)

        tk.Label(form, text="Chest Pain Type:").grid(row=9, column=0, sticky="w", pady=5)
        self.cp_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.cp_var, values=list(CP_OPTIONS.keys()), state="readonly").grid(row=9, column=1, pady=5)

        tk.Label(form, text="Resting ECG Result:").grid(row=10, column=0, sticky="w", pady=5)
        self.restecg_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.restecg_var, values=list(RESTECG_OPTIONS.keys()), state="readonly").grid(row=10, column=1, pady=5)

        tk.Label(form, text="Slope of Peak Exercise ST:").grid(row=11, column=0, sticky="w", pady=5)
        self.slope_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.slope_var, values=list(SLOPE_OPTIONS.keys()), state="readonly").grid(row=11, column=1, pady=5)

        tk.Label(form, text="Thalassemia:").grid(row=12, column=0, sticky="w", pady=5)
        self.thal_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.thal_var, values=list(THAL_OPTIONS.keys()), state="readonly").grid(row=12, column=1, pady=5)

        tk.Button(content, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(content, text="", font=("Arial", 13, "bold"), fg="black")
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
            age = float(self.age_entry.get())
            sex = SEX_MAP[self.sex_var.get()]
            trestbps = float(self.trestbps_entry.get())
            chol = float(self.chol_entry.get())
            fbs = YES_NO_MAP[self.fbs_var.get()]
            thalach = float(self.thalach_entry.get())
            exang = YES_NO_MAP[self.exang_var.get()]
            oldpeak = float(self.oldpeak_entry.get())
            ca = float(self.ca_entry.get())
            cp = CP_OPTIONS[self.cp_var.get()]
            restecg = RESTECG_OPTIONS[self.restecg_var.get()]
            slope = SLOPE_OPTIONS[self.slope_var.get()]
            thal = THAL_OPTIONS[self.thal_var.get()]
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        if not self._check_range("Resting Blood Pressure", trestbps, 94, 200):
            return
        if not self._check_range("Cholesterol", chol, 126, 564):
            return
        if not self._check_range("Max Heart Rate", thalach, 71, 202):
            return

        sample = {
            'age': age,
            'sex': sex,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'ca': ca,
            'cp_2.0': 1 if cp == 2 else 0,
            'cp_3.0': 1 if cp == 3 else 0,
            'cp_4.0': 1 if cp == 4 else 0,
            'restecg_1.0': 1 if restecg == 1 else 0,
            'restecg_2.0': 1 if restecg == 2 else 0,
            'slope_2.0': 1 if slope == 2 else 0,
            'slope_3.0': 1 if slope == 3 else 0,
            'thal_6.0': 1 if thal == 6 else 0,
            'thal_7.0': 1 if thal == 7 else 0,
        }

        bundle = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "heart_disease_model.pkl"))
        model = bundle['model']
        scaler = bundle['scaler']
        feature_columns = bundle['feature_columns']
        numeric_features = bundle['numeric_features']

        input_df = pd.DataFrame([sample])[feature_columns]

        # فقط ستون‌های عددی رو scale می‌کنیم، ستون‌های one-hot دست‌نخورده می‌مونن
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        prediction = model.predict(input_df)[0]

        result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        self.result_label.config(text=f"Prediction: {result_text}")