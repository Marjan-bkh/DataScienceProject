import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('DataSets/Reviews.csv')
# print(df.head())
# print(df.shape)
# print(df.describe())

# print(df['Score'].value_counts().sort_index())
# print(df.duplicated(subset=['UserId', 'ProfileName', 'Time', 'Text']).sum())

# حذف تکراری‌ها
df = df.drop_duplicates(subset=['UserId', 'ProfileName', 'Time', 'Text']).reset_index(drop=True)
# فیلتر کردن نمونه‌های neutral و بازساخت لیبل باینری
df_binary = df[df['Sentiment'] != 'neutral'].copy()
df_binary['Sentiment'] = df_binary['Sentiment'].map({'positive': 1, 'negative': 0})


# # ساخت لیبل سه‌کلاسه
# def to_sentiment(score):
#     if score <= 2:
#         return 'negative'
#     elif score == 3:
#         return 'neutral'
#     else:
#         return 'positive'
# df['Sentiment'] = df['Score'].apply(to_sentiment)
# print(df.shape)
# print(df['Sentiment'].value_counts())
# print(df['Sentiment'].value_counts(normalize=True))

# # چند نمونه واقعی رو ببین
# for t in df['Text'].sample(5, random_state=42):
#     print(t)
#     print('---')

# # چک کن چند تا ریویو تگ HTML دارن
# import re
# has_html = df['Text'].str.contains(r'<[^>]+>', regex=True).sum()
# print(f"تعداد ریویوهایی که تگ HTML دارن: {has_html}")

# # توزیع طول متن (بر حسب تعداد کلمه)
# df['text_len'] = df['Text'].str.split().str.len()
# print(df['text_len'].describe())

import re
contractions_map = {
    "won't": "will not", "can't": "can not", "n't": " not",
    "'re": " are", "'s": "", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am"
}
def expand_contractions(text):
    text = text.lower()
    # یکسان‌سازی انواع مختلف آپاستروف (raw string معمولی، یونیکد) به یه نوع استاندارد
    text = text.replace("’", "'").replace("`", "'")
    for pattern, repl in contractions_map.items():
        text = text.replace(pattern, repl)
    return text

def clean_text(text):
    text = expand_contractions(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z!\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['Text'].apply(clean_text)

# sample_df = df.sample(5, random_state=1)
# for orig, clean in zip(sample_df['Text'], sample_df['clean_text']):
#     print("ORIGINAL:", orig[:150])
#     print("CLEAN:", clean[:150])
#     print('---')

import nltk
# nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# کلمات نفی که نباید حذف بشن
negation_words = {'not', 'no', 'nor', 'never', 'none', 'nothing',
                   'neither', 'nowhere', 'cannot', "don't", "doesn't"}
stop_words = stop_words - negation_words

def remove_stopwords(text):
    words = text.split()
    filtered = [w for w in words if w not in stop_words]
    return ' '.join(filtered)

df['final_text'] = df['clean_text'].apply(remove_stopwords)

# sample_df = df.sample(5, random_state=1)
# for clean, final in zip(sample_df['clean_text'], sample_df['final_text']):
#     print("CLEAN:", clean[:150])
#     print("FINAL:", final[:150])
#     print('---')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# X = df['final_text']
# y = df['Sentiment']
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
# X_train_tfidf = tfidf.fit_transform(X_train)
# X_test_tfidf = tfidf.transform(X_test)
# print(X_train_tfidf.shape)
# print(X_test_tfidf.shape)

# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
#
# model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
# model.fit(X_train_tfidf, y_train)
#
# y_pred = model.predict(X_test_tfidf)
#
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

from sklearn.svm import LinearSVC
#
# svm_model = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)
# svm_model.fit(X_train_tfidf, y_train)
#
# y_pred_svm = svm_model.predict(X_test_tfidf)
# print(classification_report(y_test, y_pred_svm))
# print(confusion_matrix(y_test, y_pred_svm))

from sklearn.model_selection import GridSearchCV

# param_grid = {'C': [0.1, 0.5, 1, 5, 10]}
# grid_search = GridSearchCV(
#     LinearSVC(class_weight='balanced', random_state=42, max_iter=5000),
#     param_grid,
#     cv=3,
#     scoring='f1_macro',
#     n_jobs=-1,
#     verbose=2
# )
# grid_search.fit(X_train_tfidf, y_train)
# print("Best C:", grid_search.best_params_)
# print("Best macro F1 (CV):", grid_search.best_score_)
#
# best_model = grid_search.best_estimator_
# y_pred_best = best_model.predict(X_test_tfidf)
# print(classification_report(y_test, y_pred_best))

#
# tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2)
# X_train_tfidf = tfidf.fit_transform(X_train)
# X_test_tfidf = tfidf.transform(X_test)
#
# print(X_train_tfidf.shape)
#
# final_model = LinearSVC(C=0.1, class_weight='balanced', random_state=42, max_iter=5000)
# final_model.fit(X_train_tfidf, y_train)
#
# y_pred_final = final_model.predict(X_test_tfidf)
# print(classification_report(y_test, y_pred_final))

