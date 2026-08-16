import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import os
from scrollable_frame import ScrollableFrame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GLASS_TYPE_NAMES = {
    1: "Building Windows (Float Processed)",
    2: "Building Windows (Non-Float Processed)",
    3: "Vehicle Windows (Float Processed)",
    5: "Containers",
    6: "Tableware",
    7: "Headlamps",
}


class PredictionPageGlass(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        scroll_container = ScrollableFrame(self)
        scroll_container.pack(fill="both", expand=True)
        content = scroll_container.scrollable_frame

        tk.Label(content, text="Glass Type Identification", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Identifies the type of glass (e.g. window, container, headlamp) based on its "
            "chemical composition.\nModel: Random Forest Classifier."
        )
        tk.Label(content, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(content, text="Model Performance (Accuracy): 81.4%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(content)
        form.pack(pady=10)

        fields = [
            ("ri", "Refractive Index (1.5112-1.5339):"),
            ("na", "Sodium / Na (10.7-17.4):"),
            ("mg", "Magnesium / Mg (0-4.5):"),
            ("al", "Aluminum / Al (0.3-3.5):"),
            ("si", "Silicon / Si (69.8-75.4):"),
            ("k", "Potassium / K (0-6.2):"),
            ("ca", "Calcium / Ca (5.4-16.2):"),
            ("ba", "Barium / Ba (0-3.2):"),
            ("fe", "Iron / Fe (0-0.5):"),
        ]

        self.entries = {}
        for i, (key, label) in enumerate(fields):
            tk.Label(form, text=label).grid(row=i, column=0, sticky="w", pady=5)
            entry = tk.Entry(form)
            entry.grid(row=i, column=1, pady=5)
            self.entries[key] = entry

        tk.Button(content, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(content, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            ri = float(self.entries["ri"].get())
            na = float(self.entries["na"].get())
            mg = float(self.entries["mg"].get())
            al = float(self.entries["al"].get())
            si = float(self.entries["si"].get())
            k = float(self.entries["k"].get())
            ca = float(self.entries["ca"].get())
            ba = float(self.entries["ba"].get())
            fe = float(self.entries["fe"].get())
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {
            'RI': ri, 'Na': na, 'Mg': mg, 'Al': al, 'Si': si,
            'K': k, 'Ca': ca, 'Ba': ba, 'Fe': fe,
            'Ca_Na_ratio': ca / na,
            'Al_Si_ratio': al / si,
            'Total_modifiers': na + mg + ca + k,
            'has_Ba': 1 if ba > 0 else 0,
            'has_Fe': 1 if fe > 0 else 0,
        }

        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "glass_type_rf_model.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "glass_model_features.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        result_text = GLASS_TYPE_NAMES.get(prediction, f"Type {prediction}")
        self.result_label.config(text=f"Predicted Glass Type: {result_text}")