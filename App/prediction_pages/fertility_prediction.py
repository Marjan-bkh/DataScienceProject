import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from App.scrollable_frame import ScrollableFrame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEASON_MAP = {"Winter": -1, "Spring": -0.33, "Summer": 0.33, "Fall": 1}
YES_NO_MAP = {"No": 0, "Yes": 1}
FEVER_MAP = {"Less than 3 months ago": -1, "More than 3 months ago": 0, "No": 1}
ALCOHOL_MAP = {
    "Several times a day": 0.2,
    "Every day": 0.4,
    "Several times a week": 0.6,
    "Once a week": 0.8,
    "Hardly ever or never": 1.0,
}
SMOKING_MAP = {"Never": -1, "Occasional": 0, "Daily": 1}


class PredictionPageFertility(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        scroll_container = ScrollableFrame(self)
        scroll_container.pack(fill="both", expand=True)
        content = scroll_container.scrollable_frame

        tk.Label(content, text="Fertility Diagnosis Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts semen quality diagnosis (Normal / Altered) based on a lifestyle "
            "and health questionnaire, per WHO 2010 criteria.\nModel: Logistic Regression."
        )
        tk.Label(content, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(content, text="Model Performance (F1-Macro): 49%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(content)
        form.pack(pady=10)

        row = 0

        tk.Label(form, text="Season:").grid(row=row, column=0, sticky="w", pady=4)
        self.season_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.season_var, values=list(SEASON_MAP.keys()), state="readonly", width=20).grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Age (18-36):").grid(row=row, column=0, sticky="w", pady=4)
        self.age_entry = tk.Entry(form, width=22)
        self.age_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Childish Diseases (chicken pox, measles, mumps, polio):", wraplength=350, justify="left").grid(row=row, column=0, sticky="w", pady=4)
        self.childish_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.childish_var, values=list(YES_NO_MAP.keys()), state="readonly", width=20).grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Accident or Serious Trauma:").grid(row=row, column=0, sticky="w", pady=4)
        self.trauma_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.trauma_var, values=list(YES_NO_MAP.keys()), state="readonly", width=20).grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Surgical Intervention:").grid(row=row, column=0, sticky="w", pady=4)
        self.surgery_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.surgery_var, values=list(YES_NO_MAP.keys()), state="readonly", width=20).grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="High Fevers in the Last Year:").grid(row=row, column=0, sticky="w", pady=4)
        self.fever_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.fever_var, values=list(FEVER_MAP.keys()), state="readonly", width=20).grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Frequency of Alcohol Consumption:").grid(row=row, column=0, sticky="w", pady=4)
        self.alcohol_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.alcohol_var, values=list(ALCOHOL_MAP.keys()), state="readonly", width=20).grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Smoking Habit:").grid(row=row, column=0, sticky="w", pady=4)
        self.smoking_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.smoking_var, values=list(SMOKING_MAP.keys()), state="readonly", width=20).grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Hours Sitting Per Day (0-16):").grid(row=row, column=0, sticky="w", pady=4)
        self.hours_entry = tk.Entry(form, width=22)
        self.hours_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Button(content, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(content, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            season = SEASON_MAP[self.season_var.get()]

            age_raw = float(self.age_entry.get())
            if not (18 <= age_raw <= 36):
                raise ValueError
            age = (age_raw - 18) / 18

            childish = YES_NO_MAP[self.childish_var.get()]
            trauma = YES_NO_MAP[self.trauma_var.get()]
            surgery = YES_NO_MAP[self.surgery_var.get()]
            fever = FEVER_MAP[self.fever_var.get()]
            alcohol = ALCOHOL_MAP[self.alcohol_var.get()]
            smoking = SMOKING_MAP[self.smoking_var.get()]

            hours_raw = float(self.hours_entry.get())
            if not (0 <= hours_raw <= 16):
                raise ValueError
            hours = hours_raw / 16

        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields with valid values.")
            return

        sample = {
            "Season": season,
            "Age": age,
            "Childish_Diseases": childish,
            "Accident_Trauma": trauma,
            "Surgical_Intervention": surgery,
            "High_Fevers_LastYear": fever,
            "Alcohol_Consumption": alcohol,
            "Smoking_Habit": smoking,
            "Hours_Sitting_PerDay": hours,
        }

        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "fertility_lr_model.pkl"))
        feature_columns = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "fertility_feature_columns.pkl"))

        input_df = pd.DataFrame([sample])[feature_columns]
        prediction = model.predict(input_df)[0]

        result_text = "Altered (O)" if prediction == 1 else "Normal (N)"
        self.result_label.config(text=f"Diagnosis: {result_text}")