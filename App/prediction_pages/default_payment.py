import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from scrollable_frame import ScrollableFrame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEX_MAP = {"Male": 1, "Female": 2}
EDUCATION_OPTIONS = ["Graduate School (baseline)", "University", "High School", "Other"]
EDUCATION_COL_MAP = {"University": "EDUCATION_2", "High School": "EDUCATION_3", "Other": "EDUCATION_4"}
MARRIAGE_OPTIONS = ["Married (baseline)", "Single", "Other"]
MARRIAGE_COL_MAP = {"Single": "MARRIAGE_2", "Other": "MARRIAGE_3"}

PAY_STATUS_HELP = "(-2=No consumption, -1=Paid in full, 0=Revolving credit, 1-8=Months delayed)"


class PredictionPageCreditDefault(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        scroll_container = ScrollableFrame(self)
        scroll_container.pack(fill="both", expand=True)
        content = scroll_container.scrollable_frame

        tk.Label(content, text="Credit Card Default Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts whether a credit card client will default on their payment next "
            "month, based on billing history and repayment behavior.\nModel: Random Forest Classifier."
        )
        tk.Label(content, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(content, text="Model Performance (F1-Score): 51.2%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(content)
        form.pack(pady=10)

        row = 0
        tk.Label(form, text="Credit Limit ($):").grid(row=row, column=0, sticky="w", pady=4)
        self.limit_entry = tk.Entry(form)
        self.limit_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Sex:").grid(row=row, column=0, sticky="w", pady=4)
        self.sex_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.sex_var, values=list(SEX_MAP.keys()), state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Age:").grid(row=row, column=0, sticky="w", pady=4)
        self.age_entry = tk.Entry(form)
        self.age_entry.grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Education:").grid(row=row, column=0, sticky="w", pady=4)
        self.education_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.education_var, values=EDUCATION_OPTIONS, state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text="Marital Status:").grid(row=row, column=0, sticky="w", pady=4)
        self.marriage_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.marriage_var, values=MARRIAGE_OPTIONS, state="readonly").grid(row=row, column=1, pady=4)
        row += 1

        tk.Label(form, text=f"Repayment Status (most recent month) {PAY_STATUS_HELP}", wraplength=350, justify="left").grid(row=row, column=0, sticky="w", pady=4)
        row += 1
        self.pay_entries = []
        pay_labels = ["Month -1 (most recent)", "Month -2", "Month -3", "Month -4", "Month -5", "Month -6"]
        for label in pay_labels:
            tk.Label(form, text=f"  {label}:").grid(row=row, column=0, sticky="w", pady=2)
            entry = tk.Entry(form)
            entry.grid(row=row, column=1, pady=2)
            self.pay_entries.append(entry)
            row += 1

        tk.Label(form, text="Bill Amounts (last 6 months, $):").grid(row=row, column=0, sticky="w", pady=4)
        row += 1
        self.bill_entries = []
        for i in range(1, 7):
            tk.Label(form, text=f"  Month -{i}:").grid(row=row, column=0, sticky="w", pady=2)
            entry = tk.Entry(form)
            entry.grid(row=row, column=1, pady=2)
            self.bill_entries.append(entry)
            row += 1

        tk.Label(form, text="Payment Amounts (last 6 months, $):").grid(row=row, column=0, sticky="w", pady=4)
        row += 1
        self.payamt_entries = []
        for i in range(1, 7):
            tk.Label(form, text=f"  Month -{i}:").grid(row=row, column=0, sticky="w", pady=2)
            entry = tk.Entry(form)
            entry.grid(row=row, column=1, pady=2)
            self.payamt_entries.append(entry)
            row += 1

        tk.Button(content, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(content, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            limit_bal = float(self.limit_entry.get())
            sex = SEX_MAP[self.sex_var.get()]
            age = float(self.age_entry.get())
            education = self.education_var.get()
            marriage = self.marriage_var.get()
            if not education or not marriage:
                raise ValueError

            pay_status = [float(e.get()) for e in self.pay_entries]
            bill_amts = [float(e.get()) for e in self.bill_entries]
            pay_amts = [float(e.get()) for e in self.payamt_entries]
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'LIMIT_BAL': limit_bal,
            'SEX': sex,
            'AGE': age,
            'PAY_0': pay_status[0], 'PAY_2': pay_status[1], 'PAY_3': pay_status[2],
            'PAY_4': pay_status[3], 'PAY_5': pay_status[4], 'PAY_6': pay_status[5],
            'BILL_AMT1': bill_amts[0], 'BILL_AMT2': bill_amts[1], 'BILL_AMT3': bill_amts[2],
            'BILL_AMT4': bill_amts[3], 'BILL_AMT5': bill_amts[4], 'BILL_AMT6': bill_amts[5],
            'PAY_AMT1': pay_amts[0], 'PAY_AMT2': pay_amts[1], 'PAY_AMT3': pay_amts[2],
            'PAY_AMT4': pay_amts[3], 'PAY_AMT5': pay_amts[4], 'PAY_AMT6': pay_amts[5],
            'avg_bill_amt': sum(bill_amts) / 6,
            'bill_trend': bill_amts[0] - bill_amts[5],
            'avg_pay_amt': sum(pay_amts) / 6,
            'avg_pay_status': sum(pay_status) / 6,
        }

        for col in EDUCATION_COL_MAP.values():
            sample[col] = 0
        if education in EDUCATION_COL_MAP:
            sample[EDUCATION_COL_MAP[education]] = 1

        for col in MARRIAGE_COL_MAP.values():
            sample[col] = 0
        if marriage in MARRIAGE_COL_MAP:
            sample[MARRIAGE_COL_MAP[marriage]] = 1

        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "default_payment_rf_model.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "default_payment_features.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        result_text = "Likely to Default" if prediction == 1 else "Not Likely to Default"
        self.result_label.config(text=f"Prediction: {result_text}")