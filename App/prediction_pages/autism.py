import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from App.scrollable_frame import ScrollableFrame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

YES_NO_MAP = {"No": 0, "Yes": 1}

A_QUESTIONS = [
    ("A1_Score", "I often notice small sounds when others do not"),
    ("A2_Score", "I usually concentrate more on the whole picture, rather than the small details"),
    ("A3_Score", "I find it easy to do more than one thing at once"),
    ("A4_Score", "If there is an interruption, I can switch back to what I was doing quickly"),
    ("A5_Score", "I find it easy to 'read between the lines' when someone is talking to me"),
    ("A6_Score", "I know how to tell if someone listening to me is getting bored"),
    ("A7_Score", "I find it easy to work out what someone is thinking or feeling just by looking at their face"),
    ("A8_Score", "I find it difficult to work out people's intentions"),
    ("A9_Score", "I find it easy to work out what someone is thinking or feeling by their voice"),
    ("A10_Score", "I find it hard to make new friends"),
]

# alphabetically-first category was dropped as baseline in each group
ETHNICITY_OPTIONS = ["Asian (baseline)", "Black", "Hispanic", "Latino", "Middle Eastern",
                     "Others", "Pasifika", "South Asian", "Turkish", "Unknown", "White-European"]
ETHNICITY_COL_MAP = {
    "Black": "ethnicity_Black", "Hispanic": "ethnicity_Hispanic", "Latino": "ethnicity_Latino",
    "Middle Eastern": "ethnicity_Middle Eastern ", "Others": "ethnicity_Others",
    "Pasifika": "ethnicity_Pasifika", "South Asian": "ethnicity_South Asian",
    "Turkish": "ethnicity_Turkish", "Unknown": "ethnicity_Unknown",
    "White-European": "ethnicity_White-European",
}

COUNTRY_OPTIONS = ["India (baseline)", "New Zealand", "United Arab Emirates", "United Kingdom", "United States", "Other"]
COUNTRY_COL_MAP = {
    "New Zealand": "contry_of_res_New Zealand", "United Arab Emirates": "contry_of_res_United Arab Emirates",
    "United Kingdom": "contry_of_res_United Kingdom", "United States": "contry_of_res_United States",
    "Other": "contry_of_res_Other",
}

RELATION_OPTIONS = ["Health care professional (baseline)", "Others", "Parent", "Relative", "Self", "Unknown"]
RELATION_COL_MAP = {
    "Others": "relation_Others", "Parent": "relation_Parent", "Relative": "relation_Relative",
    "Self": "relation_Self", "Unknown": "relation_Unknown",
}


class PredictionPageAutism(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        scroll_container = ScrollableFrame(self)
        scroll_container.pack(fill="both", expand=True)
        content = scroll_container.scrollable_frame

        tk.Label(content, text="Autism Spectrum Screening (Adult)", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Screens for likelihood of Autism Spectrum traits based on the AQ-10 "
            "questionnaire and demographic information.\nModel: Logistic Regression."
        )
        tk.Label(content, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(content, text="Model Performance (F1-Score): 98.7%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(content)
        form.pack(pady=10)

        row = 0
        self.a_vars = {}
        for key, question in A_QUESTIONS:
            tk.Label(form, text=question, wraplength=350, justify="left").grid(row=row, column=0, sticky="w", pady=4)
            var = tk.StringVar()
            ttk.Combobox(form, textvariable=var, values=list(YES_NO_MAP.keys()), state="readonly", width=8).grid(row=row, column=1, pady=4)
            self.a_vars[key] = var
            row += 1

        tk.Label(form, text="Age:").grid(row=row, column=0, sticky="w", pady=4)
        self.age_entry = tk.Entry(form)
        self.age_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Gender:").grid(row=row, column=0, sticky="w", pady=4)
        self.gender_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.gender_var, values=["Male", "Female"], state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="History of Jaundice at Birth:").grid(row=row, column=0, sticky="w", pady=4)
        self.jaundice_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.jaundice_var, values=list(YES_NO_MAP.keys()), state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Family Member with Autism:").grid(row=row, column=0, sticky="w", pady=4)
        self.austim_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.austim_var, values=list(YES_NO_MAP.keys()), state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Used Screening App Before:").grid(row=row, column=0, sticky="w", pady=4)
        self.used_app_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.used_app_var, values=list(YES_NO_MAP.keys()), state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Ethnicity:").grid(row=row, column=0, sticky="w", pady=4)
        self.ethnicity_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.ethnicity_var, values=ETHNICITY_OPTIONS, state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Country of Residence:").grid(row=row, column=0, sticky="w", pady=4)
        self.country_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.country_var, values=COUNTRY_OPTIONS, state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Relation to Person Completing Test:").grid(row=row, column=0, sticky="w", pady=4)
        self.relation_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.relation_var, values=RELATION_OPTIONS, state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Button(content, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(content, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            a_values = {key: YES_NO_MAP[var.get()] for key, var in self.a_vars.items()}
            age = float(self.age_entry.get())
            gender = 1 if self.gender_var.get() == "Male" else 0
            jaundice = YES_NO_MAP[self.jaundice_var.get()]
            austim = YES_NO_MAP[self.austim_var.get()]
            used_app = YES_NO_MAP[self.used_app_var.get()]
            ethnicity = self.ethnicity_var.get()
            country = self.country_var.get()
            relation = self.relation_var.get()
            if not ethnicity or not country or not relation:
                raise ValueError
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = dict(a_values)
        sample['age'] = age
        sample['gender'] = gender
        sample['jundice'] = jaundice
        sample['austim'] = austim
        sample['used_app_before'] = used_app

        for col in ETHNICITY_COL_MAP.values():
            sample[col] = 0
        if ethnicity in ETHNICITY_COL_MAP:
            sample[ETHNICITY_COL_MAP[ethnicity]] = 1

        for col in COUNTRY_COL_MAP.values():
            sample[col] = 0
        if country in COUNTRY_COL_MAP:
            sample[COUNTRY_COL_MAP[country]] = 1

        for col in RELATION_COL_MAP.values():
            sample[col] = 0
        if relation in RELATION_COL_MAP:
            sample[RELATION_COL_MAP[relation]] = 1

        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "autism_lr_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "autism_scaler.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "autism_model_features.pkl"))

        # محاسبه‌ی result (مجموع A1 تا A10) چون scaler بهش نیاز داره، هرچند مدل نهایی ازش استفاده نمی‌کنه
        sample_for_scaling = dict(sample)
        sample_for_scaling['result'] = sum(a_values.values())

        # ترتیب دقیق ستون‌هایی که scaler موقع fit دیده (شامل result)
        scaler_columns = list(scaler.feature_names_in_)
        input_df_for_scaling = pd.DataFrame([sample_for_scaling])[scaler_columns]
        input_scaled = scaler.transform(input_df_for_scaling)
        input_scaled_df = pd.DataFrame(input_scaled, columns=scaler_columns)

        # حالا فقط ستون‌هایی که مدل واقعاً استفاده می‌کنه (بدون result) رو نگه می‌داریم
        input_scaled_df = input_scaled_df[feature_names]

        prediction = model.predict(input_scaled_df)[0]

        result_text = "Traits Indicative of ASD" if prediction == 1 else "No Strong Indication of Autism Disorder"
        self.result_label.config(text=f"Screening Result: {result_text}")