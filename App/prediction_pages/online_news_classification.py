import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from scrollable_frame import ScrollableFrame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_CHANNEL_MAP = {
    "Lifestyle": "data_channel_is_lifestyle",
    "Entertainment": "data_channel_is_entertainment",
    "Business": "data_channel_is_bus",
    "Social Media": "data_channel_is_socmed",
    "Tech": "data_channel_is_tech",
    "World": "data_channel_is_world",
}

WEEKDAY_MAP = {
    "Monday": "weekday_is_monday", "Tuesday": "weekday_is_tuesday",
    "Wednesday": "weekday_is_wednesday", "Thursday": "weekday_is_thursday",
    "Friday": "weekday_is_friday", "Saturday": "weekday_is_saturday",
    "Sunday": "weekday_is_sunday",
}

MEDIAN_DEFAULTS = {
    'n_unique_tokens': 0.5389221547475, 'n_non_stop_words': 0.999999996,
    'n_non_stop_unique_tokens': 0.69014084183, 'num_self_hrefs': 3.0,
    'average_token_length': 4.66304347826,
    'kw_min_min': -1.0, 'kw_max_min': 662.0, 'kw_avg_min': 235.97071428549998,
    'kw_min_max': 1400.0, 'kw_max_max': 843300.0, 'kw_avg_max': 245042.8357145,
    'kw_min_avg': 1030.9864864850001, 'kw_max_avg': 4361.616601435, 'kw_avg_avg': 2874.483307305,
    'self_reference_min_shares': 1200.0, 'self_reference_max_shares': 2900.0,
    'self_reference_avg_sharess': 2209.125,
    'LDA_00': 0.03338991189705, 'LDA_01': 0.0333446755678, 'LDA_02': 0.0400031407642,
    'LDA_03': 0.0400007008651, 'LDA_04': 0.040708539787849995,
    'global_rate_positive_words': 0.0391459074733, 'global_rate_negative_words': 0.0153139356815,
    'rate_positive_words': 0.712121212121, 'rate_negative_words': 0.277777777778,
    'avg_positive_polarity': 0.3587184214595, 'min_positive_polarity': 0.1, 'max_positive_polarity': 0.8,
    'avg_negative_polarity': -0.25339023919750003, 'min_negative_polarity': -0.5, 'max_negative_polarity': -0.1,
}

SUBJECTIVITY_MAP = {
    "Very Factual": 0.1,
    "Mostly Factual": 0.3,
    "Balanced": 0.5,
    "Mostly Opinionated": 0.7,
    "Very Opinionated": 0.9,
}

SENTIMENT_MAP = {
    "Very Negative": -0.8,
    "Somewhat Negative": -0.4,
    "Neutral": 0.0,
    "Somewhat Positive": 0.4,
    "Very Positive": 0.8,
}


class PredictionPageOnlineNewsClassification(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="classification")).pack(anchor="w", padx=15, pady=10)

        scroll_container = ScrollableFrame(self)
        scroll_container.pack(fill="both", expand=True)
        content = scroll_container.scrollable_frame

        tk.Label(content, text="Online News Popularity (Viral or Not)", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts whether an online news article will go viral (1400+ shares) based "
            "on content structure, topic, and tone. Only key article features are "
            "collected here; less intuitive statistical features are filled in using "
            "dataset averages.\nModel: Gradient Boosting Classifier."
        )
        tk.Label(content, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(content, text="Model Performance (F1-Score): 69.9%", font=("Arial", 9, "italic"), fg="gray8").pack(pady=(0, 10))

        form = tk.Frame(content)
        form.pack(pady=10)

        tk.Label(form, text="Title Length (words):").grid(row=0, column=0, sticky="w", pady=5)
        self.title_len_entry = tk.Entry(form)
        self.title_len_entry.grid(row=0, column=1, pady=5)

        tk.Label(form, text="Content Length (words):").grid(row=1, column=0, sticky="w", pady=5)
        self.content_len_entry = tk.Entry(form)
        self.content_len_entry.grid(row=1, column=1, pady=5)

        tk.Label(form, text="Number of Links:").grid(row=2, column=0, sticky="w", pady=5)
        self.hrefs_entry = tk.Entry(form)
        self.hrefs_entry.grid(row=2, column=1, pady=5)

        tk.Label(form, text="Number of Images:").grid(row=3, column=0, sticky="w", pady=5)
        self.imgs_entry = tk.Entry(form)
        self.imgs_entry.grid(row=3, column=1, pady=5)

        tk.Label(form, text="Number of Videos:").grid(row=4, column=0, sticky="w", pady=5)
        self.videos_entry = tk.Entry(form)
        self.videos_entry.grid(row=4, column=1, pady=5)

        tk.Label(form, text="Number of Keywords:").grid(row=5, column=0, sticky="w", pady=5)
        self.keywords_entry = tk.Entry(form)
        self.keywords_entry.grid(row=5, column=1, pady=5)

        tk.Label(form, text="Data Channel:").grid(row=6, column=0, sticky="w", pady=5)
        self.channel_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.channel_var, values=list(DATA_CHANNEL_MAP.keys()), state="readonly").grid(row=6, column=1, pady=5)

        tk.Label(form, text="Publish Day:").grid(row=7, column=0, sticky="w", pady=5)
        self.weekday_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.weekday_var, values=list(WEEKDAY_MAP.keys()), state="readonly").grid(row=7, column=1, pady=5)

        tk.Label(form, text="Content Subjectivity:").grid(row=8, column=0, sticky="w", pady=5)
        self.subjectivity_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.subjectivity_var, values=list(SUBJECTIVITY_MAP.keys()),
                     state="readonly").grid(row=8, column=1, pady=5)

        tk.Label(form, text="Content Sentiment:").grid(row=9, column=0, sticky="w", pady=5)
        self.sentiment_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.sentiment_var, values=list(SENTIMENT_MAP.keys()), state="readonly").grid(
            row=9, column=1, pady=5)

        tk.Label(form, text="Title Subjectivity:").grid(row=10, column=0, sticky="w", pady=5)
        self.title_subjectivity_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.title_subjectivity_var, values=list(SUBJECTIVITY_MAP.keys()),
                     state="readonly").grid(row=10, column=1, pady=5)

        tk.Label(form, text="Title Sentiment:").grid(row=11, column=0, sticky="w", pady=5)
        self.title_sentiment_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.title_sentiment_var, values=list(SENTIMENT_MAP.keys()),
                     state="readonly").grid(row=11, column=1, pady=5)

        tk.Button(content, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(content, text="", font=("Arial", 13, "bold"), fg="black")
        self.result_label.pack(pady=10)

    def on_show(self):
        pass

    def on_predict(self):
        try:
            title_len = float(self.title_len_entry.get())
            content_len = float(self.content_len_entry.get())
            hrefs = float(self.hrefs_entry.get())
            imgs = float(self.imgs_entry.get())
            videos = float(self.videos_entry.get())
            keywords = float(self.keywords_entry.get())
            channel_col = DATA_CHANNEL_MAP[self.channel_var.get()]
            weekday_col = WEEKDAY_MAP[self.weekday_var.get()]
            subjectivity = SUBJECTIVITY_MAP[self.subjectivity_var.get()]
            sentiment = SENTIMENT_MAP[self.sentiment_var.get()]
            title_subjectivity = SUBJECTIVITY_MAP[self.title_subjectivity_var.get()]
            title_sentiment = SENTIMENT_MAP[self.title_sentiment_var.get()]
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        sample = dict(MEDIAN_DEFAULTS)
        sample['n_tokens_title'] = title_len
        sample['n_tokens_content'] = content_len
        sample['num_hrefs'] = hrefs
        sample['num_imgs'] = imgs
        sample['num_videos'] = videos
        sample['num_keywords'] = keywords
        sample['global_subjectivity'] = subjectivity
        sample['global_sentiment_polarity'] = sentiment
        sample['title_subjectivity'] = title_subjectivity
        sample['title_sentiment_polarity'] = title_sentiment
        sample['abs_title_subjectivity'] = abs(title_subjectivity - 0.5)
        sample['abs_title_sentiment_polarity'] = abs(title_sentiment)

        for col in DATA_CHANNEL_MAP.values():
            sample[col] = 1 if col == channel_col else 0

        for col in WEEKDAY_MAP.values():
            sample[col] = 1 if col == weekday_col else 0
        sample['is_weekend'] = 1 if weekday_col in ("weekday_is_saturday", "weekday_is_sunday") else 0

        model = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "online_news_popularity_gb_model.pkl"))
        feature_names = joblib.load(os.path.join(BASE_DIR, "Classification", "ModelsOutcome", "online_news_popularity_feature_columns.pkl"))

        input_df = pd.DataFrame([sample])[feature_names]
        prediction = model.predict(input_df)[0]

        result_text = "Likely to Go Viral (1400+ shares)" if prediction == 1 else "Not Likely to Go Viral"
        self.result_label.config(text=f"Prediction: {result_text}")