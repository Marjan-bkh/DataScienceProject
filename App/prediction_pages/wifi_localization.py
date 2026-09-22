import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PredictionPageWifiLocalization(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="WiFi Indoor Localization", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts which of 4 rooms a device is in, based on the signal strength "
            "(in dBm) it detects from 7 WiFi access points.\nModel: Logistic Regression."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (Accuracy): 98.0%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        self.wifi_entries = []
        for i in range(1, 8):
            tk.Label(form, text=f"WiFi Signal {i} (dBm):").grid(row=i - 1, column=0, sticky="w", pady=5)
            entry = tk.Entry(form)
            entry.grid(row=i - 1, column=1, pady=5)
            self.wifi_entries.append(entry)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            wifi_values = [float(e.get()) for e in self.wifi_entries]
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        # همون مهندسی ویژگی زمان train، دقیقاً به همون ترتیب
        sorted_values = sorted(wifi_values)
        strongest_signal = sorted_values[-1]
        signal_range = sorted_values[-1] - sorted_values[0]
        strongest_router = wifi_values.index(max(wifi_values))

        sample = {
            'wifi_1': wifi_values[0],
            'wifi_2': wifi_values[1],
            'wifi_3': wifi_values[2],
            'wifi_4': wifi_values[3],
            'wifi_5': wifi_values[4],
            'wifi_6': wifi_values[5],
            'wifi_7': wifi_values[6],
            'strongest_signal': strongest_signal,
            'signal_range': signal_range,
            'strongest_router': strongest_router,
        }

        bundle = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "wifi_room_model.pkl"))
        model = bundle['model']
        feature_names = bundle['feature_names']
        scaler = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "wifi_room_scaler.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        self.result_label.config(text=f"Predicted Room: {int(prediction)}")