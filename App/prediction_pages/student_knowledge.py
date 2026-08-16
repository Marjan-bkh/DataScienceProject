import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLUSTER_LABELS = {
    1: "Very High Knowledge",
    3: "High Knowledge",
    4: "Middle Knowledge",
    0: "Low Knowledge",
    2: "Very Low Knowledge",
}


class PredictionPageStudentKnowledge(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="clustering")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Student Knowledge Level Clustering", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Groups students into knowledge-level clusters based on study time and exam "
            "performance patterns (Electrical DC Machines course).\n"
            "Model: Gaussian Mixture Model (5 components)."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Cluster-to-knowledge-level mapping is approximate (ARI ≈ 0.26 vs. true labels).",
                 font=("Arial", 9, "italic"), fg="gray8", wraplength=550, justify="center").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Study Time for Goal Materials (0-1):").grid(row=0, column=0, sticky="w", pady=5)
        self.stg_entry = tk.Entry(form)
        self.stg_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Repetition Number for Goal Materials (0-1):").grid(row=1, column=0, sticky="w", pady=5)
        self.scg_entry = tk.Entry(form)
        self.scg_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="Study Time for Related Objects (0-1):").grid(row=2, column=0, sticky="w", pady=5)
        self.str_entry = tk.Entry(form)
        self.str_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Exam Performance for Related Objects / LPR (0-1):").grid(row=3, column=0, sticky="w", pady=5)
        self.lpr_entry = tk.Entry(form)
        self.lpr_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Exam Performance for Goal Objects / PEG (0-1):").grid(row=4, column=0, sticky="w", pady=5)
        self.peg_entry = tk.Entry(form)
        self.peg_entry.grid(row=4, column=1, pady=5)

        tk.Button(self, text="Find My Knowledge Level", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black", wraplength=550, justify="center")
        self.result_label.pack(pady=10, padx=20)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            v1 = float(self.stg_entry.get())
            v2 = float(self.scg_entry.get())
            v3 = float(self.str_entry.get())
            v4 = float(self.lpr_entry.get())
            v5 = float(self.peg_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5}

        model = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "students_knowledge_gmm_final_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "students_knowledge_scaler.pkl"))
        feature_names = list(scaler.feature_names_in_)

        input_df = pd.DataFrame([sample])[feature_names]
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

        cluster = model.predict(input_scaled_df)[0]
        label = CLUSTER_LABELS.get(cluster, f"Cluster {cluster}")

        self.result_label.config(text=f"Estimated Knowledge Level: {label}")