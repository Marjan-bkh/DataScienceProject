import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CATEGORY_LABELS = [
    ("cat1", "Art Galleries"),
    ("cat2", "Dance Clubs"),
    ("cat3", "Juice Bars"),
    ("cat4", "Restaurants"),
    ("cat5", "Museums"),
    ("cat6", "Resorts"),
    ("cat8", "Beaches"),
    ("cat9", "Theaters"),
    ("cat10", "Religious Institutions"),
]

FEATURE_MAP = {
    "cat1": "Category 1", "cat2": "Category 2", "cat3": "Category 3",
    "cat4": "Category 4", "cat5": "Category 5", "cat6": "Category 6",
    "cat8": "Category 8", "cat9": "Category 9", "cat10": "Category 10",
}

CLUSTER_LABELS = {
    1: "Enthusiastic Reviewer — rates most venues highly, especially restaurants.",
    0: "Nightlife & Entertainment Enthusiast — favors dance clubs relatively more.",
    2: "Relaxation & Culture Enthusiast — favors juice bars, resorts, and museums relatively more.",
}


class PredictionPageTravelReview(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="clustering")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Traveler Type Clustering (TripAdvisor Reviews)", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Groups travelers into types based on their average ratings across venue "
            "categories.\nModel: K-Means Clustering (k=3)."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Enter your average rating (0-4 scale) for each venue category.",
                 font=("Arial", 9, "italic"), fg="gray8", wraplength=550, justify="center").pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(pady=10)

        self.entries = {}
        for i, (key, label) in enumerate(CATEGORY_LABELS):
            tk.Label(form, text=f"{label} (0-4):").grid(row=i, column=0, sticky="w", pady=5)
            entry = tk.Entry(form)
            entry.grid(row=i, column=1, pady=5)
            self.entries[key] = entry

        tk.Button(self, text="Find My Traveler Type", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black", wraplength=550, justify="center")
        self.result_label.pack(pady=10, padx=20)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            values = {key: float(entry.get()) for key, entry in self.entries.items()}
        except ValueError:
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = {FEATURE_MAP[key]: value for key, value in values.items()}

        model = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "trip_adviser_kmeans_clusters.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "trip_adviser_scaler_clusters.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Clustering", "ModelsOutcome", "trip_adviser_feature_columns.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

        cluster = model.predict(input_scaled_df)[0]
        label = CLUSTER_LABELS.get(cluster, f"Cluster {cluster}")

        self.result_label.config(text=f"Traveler Type: {label}")