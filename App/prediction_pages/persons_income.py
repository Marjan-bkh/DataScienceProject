import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from scrollable_frame import ScrollableFrame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEX_MAP = {"Male": 0, "Female": 1}

WORKCLASS_OPTIONS = ["Federal-gov (baseline)", "Local-gov", "Other", "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Unknown"]
WORKCLASS_COL_MAP = {
    "Local-gov": "workclass_Local-gov", "Other": "workclass_Other", "Private": "workclass_Private",
    "Self-emp-inc": "workclass_Self-emp-inc", "Self-emp-not-inc": "workclass_Self-emp-not-inc",
    "State-gov": "workclass_State-gov", "Unknown": "workclass_Unknown",
}

MARITAL_OPTIONS = ["Divorced (baseline)", "Married-civ-spouse", "Married-spouse-absent", "Never-married", "Other", "Separated", "Widowed"]
MARITAL_COL_MAP = {
    "Married-civ-spouse": "marital-status_Married-civ-spouse", "Married-spouse-absent": "marital-status_Married-spouse-absent",
    "Never-married": "marital-status_Never-married", "Other": "marital-status_Other",
    "Separated": "marital-status_Separated", "Widowed": "marital-status_Widowed",
}

OCCUPATION_OPTIONS = ["Adm-clerical (baseline)", "Craft-repair", "Exec-managerial", "Farming-fishing",
                      "Handlers-cleaners", "Machine-op-inspct", "Other", "Other-service", "Prof-specialty",
                      "Protective-serv", "Sales", "Tech-support", "Transport-moving", "Unknown"]
OCCUPATION_COL_MAP = {
    "Craft-repair": "occupation_Craft-repair", "Exec-managerial": "occupation_Exec-managerial",
    "Farming-fishing": "occupation_Farming-fishing", "Handlers-cleaners": "occupation_Handlers-cleaners",
    "Machine-op-inspct": "occupation_Machine-op-inspct", "Other": "occupation_Other",
    "Other-service": "occupation_Other-service", "Prof-specialty": "occupation_Prof-specialty",
    "Protective-serv": "occupation_Protective-serv", "Sales": "occupation_Sales",
    "Tech-support": "occupation_Tech-support", "Transport-moving": "occupation_Transport-moving",
    "Unknown": "occupation_Unknown",
}

RELATIONSHIP_OPTIONS = ["Husband (baseline)", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"]
RELATIONSHIP_COL_MAP = {
    "Not-in-family": "relationship_Not-in-family", "Other-relative": "relationship_Other-relative",
    "Own-child": "relationship_Own-child", "Unmarried": "relationship_Unmarried", "Wife": "relationship_Wife",
}

RACE_OPTIONS = ["Asian-Pac-Islander (baseline)", "Black", "Other", "White"]
RACE_COL_MAP = {"Black": "race_Black", "Other": "race_Other", "White": "race_White"}

COUNTRY_OPTIONS = ["Mexico (baseline)", "Other", "United-States", "Unknown"]
COUNTRY_COL_MAP = {"Other": "native-country_Other", "United-States": "native-country_United-States", "Unknown": "native-country_Unknown"}

EDUCATION_MAP = {
    "Preschool": 1,
    "1st-4th grade": 2,
    "5th-6th grade": 3,
    "7th-8th grade": 4,
    "9th grade": 5,
    "10th grade": 6,
    "11th grade": 7,
    "12th grade (no diploma)": 8,
    "High School Graduate": 9,
    "Some College": 10,
    "Associate Degree (Vocational)": 11,
    "Associate Degree (Academic)": 12,
    "Bachelor's Degree": 13,
    "Master's Degree": 14,
    "Professional School (Law/Medicine)": 15,
    "Doctorate": 16,
}


class PredictionPagePersonsIncome(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        scroll_container = ScrollableFrame(self)
        scroll_container.pack(fill="both", expand=True)
        content = scroll_container.scrollable_frame

        tk.Label(content, text="Income Prediction (Adult Census)", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts whether a person's annual income exceeds $50K based on census "
            "demographic and employment data.\nModel: Random Forest Classifier."
        )
        tk.Label(content, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(content, text="Model Performance (F1-Score): 66.7%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(content)
        form.pack(pady=10)

        row = 0
        tk.Label(form, text="Age:").grid(row=row, column=0, sticky="w", pady=4)
        self.age_entry = tk.Entry(form)
        self.age_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Education Level:").grid(row=row, column=0, sticky="w", pady=4)
        self.edu_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.edu_var, values=list(EDUCATION_MAP.keys()), state="readonly").grid(row=row, column=1,pady=4)
        row += 1

        tk.Label(form, text="Sex:").grid(row=row, column=0, sticky="w", pady=4)
        self.sex_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.sex_var, values=list(SEX_MAP.keys()), state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Capital Gain ($):").grid(row=row, column=0, sticky="w", pady=4)
        self.gain_entry = tk.Entry(form)
        self.gain_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Capital Loss ($):").grid(row=row, column=0, sticky="w", pady=4)
        self.loss_entry = tk.Entry(form)
        self.loss_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Hours Worked per Week:").grid(row=row, column=0, sticky="w", pady=4)
        self.hours_entry = tk.Entry(form)
        self.hours_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Work Class:").grid(row=row, column=0, sticky="w", pady=4)
        self.workclass_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.workclass_var, values=WORKCLASS_OPTIONS, state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Marital Status:").grid(row=row, column=0, sticky="w", pady=4)
        self.marital_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.marital_var, values=MARITAL_OPTIONS, state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Occupation:").grid(row=row, column=0, sticky="w", pady=4)
        self.occupation_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.occupation_var, values=OCCUPATION_OPTIONS, state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Relationship:").grid(row=row, column=0, sticky="w", pady=4)
        self.relationship_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.relationship_var, values=RELATIONSHIP_OPTIONS, state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Race:").grid(row=row, column=0, sticky="w", pady=4)
        self.race_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.race_var, values=RACE_OPTIONS, state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Native Country:").grid(row=row, column=0, sticky="w", pady=4)
        self.country_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.country_var, values=COUNTRY_OPTIONS, state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Button(content, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(content, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            age = float(self.age_entry.get())
            edu_num = EDUCATION_MAP[self.edu_var.get()]
            sex = SEX_MAP[self.sex_var.get()]
            gain = float(self.gain_entry.get())
            loss = float(self.loss_entry.get())
            hours = float(self.hours_entry.get())
            workclass = self.workclass_var.get()
            marital = self.marital_var.get()
            occupation = self.occupation_var.get()
            relationship = self.relationship_var.get()
            race = self.race_var.get()
            country = self.country_var.get()
            if not all([workclass, marital, occupation, relationship, race, country]):
                raise ValueError
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'age': age,
            'education-num': edu_num,
            'sex': sex,
            'capital-gain': gain,
            'capital-loss': loss,
            'hours-per-week': hours,
        }

        for col in WORKCLASS_COL_MAP.values():
            sample[col] = 0
        if workclass in WORKCLASS_COL_MAP:
            sample[WORKCLASS_COL_MAP[workclass]] = 1

        for col in MARITAL_COL_MAP.values():
            sample[col] = 0
        if marital in MARITAL_COL_MAP:
            sample[MARITAL_COL_MAP[marital]] = 1

        for col in OCCUPATION_COL_MAP.values():
            sample[col] = 0
        if occupation in OCCUPATION_COL_MAP:
            sample[OCCUPATION_COL_MAP[occupation]] = 1

        for col in RELATIONSHIP_COL_MAP.values():
            sample[col] = 0
        if relationship in RELATIONSHIP_COL_MAP:
            sample[RELATIONSHIP_COL_MAP[relationship]] = 1

        for col in RACE_COL_MAP.values():
            sample[col] = 0
        if race in RACE_COL_MAP:
            sample[RACE_COL_MAP[race]] = 1

        for col in COUNTRY_COL_MAP.values():
            sample[col] = 0
        if country in COUNTRY_COL_MAP:
            sample[COUNTRY_COL_MAP[country]] = 1

        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "persons_income_rf_model.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "persons_income_columns.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        result_text = "Income > $50K" if prediction == 1 else "Income ≤ $50K"
        self.result_label.config(text=f"Prediction: {result_text}")