import os
import re
import joblib
import tkinter as tk
from tkinter import messagebox

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== پیش‌پردازش متن (باید عیناً همون مراحل train باشه) ====================
CONTRACTIONS_MAP = {
    "won't": "will not", "can't": "can not", "n't": " not",
    "'re": " are", "'s": "", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am"
}

def expand_contractions(text):
    text = text.lower()
    text = text.replace("'", "'").replace("`", "'")
    for pattern, repl in CONTRACTIONS_MAP.items():
        text = text.replace(pattern, repl)
    return text

def clean_text(text):
    text = expand_contractions(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z!\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text, stop_words):
    words = text.split()
    return ' '.join(w for w in words if w not in stop_words)


class PredictionAmazonReviewPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Back button
        tk.Button(
            self, text="← Back",
            command=lambda: controller.show_frame("category", category="classification")
        ).pack(anchor="w", padx=15, pady=10)

        # Title
        tk.Label(
            self, text="Amazon Product Reviews", font=("Arial", 16, "bold")
        ).pack(pady=(0, 5))

        # Description
        tk.Label(
            self,
            text="Predicts whether a product review expresses positive or negative sentiment.\n"
                 "Model: Linear Support Vector Classifier (TF-IDF)",
            font=("Arial", 10), fg="gray8", wraplength=550, justify="center"
        ).pack(pady=(0, 5))

        # Model Accuracy
        tk.Label(
            self, text="Model Accuracy: 92%",
            font=("Arial", 10, "italic"), fg="gray8"
        ).pack(pady=(0, 15))

        # Form
        form = tk.Frame(self)
        form.pack(pady=10)

        tk.Label(form, text="Review Text:").grid(row=0, column=0, sticky="nw", padx=5, pady=5)
        self.review_text = tk.Text(form, width=60, height=8)
        self.review_text.grid(row=0, column=1, padx=5, pady=5)

        # Predict button
        tk.Button(
            self, text="Predict", font=("Arial", 12, "bold"),
            command=self.on_predict
        ).pack(pady=10)

        # Result
        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self, dataset=None):
        pass

    def on_predict(self):
        raw_text = self.review_text.get("1.0", tk.END).strip()

        if not raw_text:
            messagebox.showerror("Error", "Please enter a review text.")
            return

        bundle = joblib.load(os.path.join(BASE_DIR, "TextAnalysis", "ModelsOutcome", "amazon_reviews_model.pkl"))
        model = bundle['model']
        tfidf = bundle['tfidf']
        stop_words = bundle['stop_words']

        cleaned = clean_text(raw_text)
        final = remove_stopwords(cleaned, stop_words)

        input_vector = tfidf.transform([final])
        prediction = model.predict(input_vector)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        self.result_label.config(text=f"Predicted Sentiment: {sentiment}")


