import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

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

# میانه‌ی فیچرهایی که مستقیم از کاربر نمی‌گیریم (از داده‌ی train)
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


class PredictionPageOnlineNews(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(self, text="← Back", command=lambda: controller.show_frame("category", category="regression")).pack(anchor="w", padx=15, pady=10)

        tk.Label(self, text="Online News Popularity (Shares) Prediction", font=("Arial", 16, "bold")).pack(pady=10)

        description_text = (
            "Predicts the number of social shares an online news article will receive, based on "
            "content structure, topic, and tone. Only key article features are collected here; "
            "less intuitive statistical features are filled in using dataset averages.\n"
            "Model: Gradient Boosting Regressor."
        )
        tk.Label(self, text=description_text, font=("Arial", 10), wraplength=550, justify="center", fg="gray8").pack(pady=(0, 5))
        tk.Label(self, text="Model Performance (R²): 16.4% — predicting viral content is inherently very noisy.",
                 font=("Arial", 9, "italic"), fg="gray8", wraplength=550, justify="center").pack(pady=(0, 10))

        form = tk.Frame(self)
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

        tk.Label(form, text="Content Subjectivity (0=Factual, 1=Opinionated):").grid(row=8, column=0, sticky="w", pady=5)
        self.subjectivity_entry = tk.Entry(form)
        self.subjectivity_entry.grid(row=8, column=1, pady=5)

        tk.Label(form, text="Content Sentiment (-1=Negative, 1=Positive):").grid(row=9, column=0, sticky="w", pady=5)
        self.sentiment_entry = tk.Entry(form)
        self.sentiment_entry.grid(row=9, column=1, pady=5)

        tk.Label(form, text="Title Subjectivity (0=Factual, 1=Opinionated):").grid(row=10, column=0, sticky="w", pady=5)
        self.title_subjectivity_entry = tk.Entry(form)
        self.title_subjectivity_entry.grid(row=10, column=1, pady=5)

        tk.Label(form, text="Title Sentiment (-1=Negative, 1=Positive):").grid(row=11, column=0, sticky="w", pady=5)
        self.title_sentiment_entry = tk.Entry(form)
        self.title_sentiment_entry.grid(row=11, column=1, pady=5)

        tk.Button(self, text="Predict", font=("Arial", 12, "bold"), command=self.on_predict).pack(pady=15)

        self.result_label = tk.Label(self, text="", font=("Arial", 13, "bold"), fg="black")
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
            subjectivity = float(self.subjectivity_entry.get())
            sentiment = float(self.sentiment_entry.get())
            title_subjectivity = float(self.title_subjectivity_entry.get())
            title_sentiment = float(self.title_sentiment_entry.get())
        except (ValueError, KeyError):
            messagebox.showerror("Input Error", "Please complete all fields.")
            return

        # از میانه‌های train شروع می‌کنیم
        sample = dict(MEDIAN_DEFAULTS)

        # فیچرهای مستقیم از کاربر
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

        # one-hot برای data channel (همه صفر، فقط انتخاب‌شده یک)
        for col in DATA_CHANNEL_MAP.values():
            sample[col] = 1 if col == channel_col else 0

        # one-hot برای روز هفته
        for col in WEEKDAY_MAP.values():
            sample[col] = 1 if col == weekday_col else 0
        sample['is_weekend'] = 1 if weekday_col in ("weekday_is_saturday", "weekday_is_sunday") else 0

        model = joblib.load("../../Regression/ModelsOutcome/online_news_popularity_gb_regressor.pkl")
        feature_names = joblib.load("../../Regression/ModelsOutcome/online_news_popularity_reg_feature_columns.pkl")

        input_df = pd.DataFrame([sample])[feature_names]
        prediction_log = model.predict(input_df)[0]

        # مدل روی log1p(shares) train شده بود، پس باید برگردونیم به عدد واقعی
        import numpy as np
        prediction_shares = np.expm1(prediction_log)

        self.result_label.config(text=f"Predicted Shares: {round(prediction_shares)}")