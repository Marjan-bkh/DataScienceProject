import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

# nltk.download('stopwords', quiet=True)

df = pd.read_csv('DataSets/Reviews.csv')
df = df.drop_duplicates(subset=['UserId', 'ProfileName', 'Time', 'Text']).reset_index(drop=True)

# ---- نسخه‌ی سه‌کلاسه (آرشیو شده، برای مرجع/گزارش) ----
# def to_sentiment(score):
#     if score <= 2:
#         return 'negative'
#     elif score == 3:
#         return 'neutral'
#     else:
#         return 'positive'
# df['Sentiment'] = df['Score'].apply(to_sentiment)
# print(df['Sentiment'].value_counts(normalize=True))
#
# X = df['final_text']
# y = df['Sentiment']
# tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
# model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
# ... (نسخه‌ی کامل سه‌کلاسه با LogisticRegression و LinearSVC قبلاً تست و مقایسه شد)
# نتیجه: macro F1 نهایی = 0.69 (محدود به‌خاطر ابهام ذاتی کلاس neutral)


df_binary = df[df['Score'] != 3].copy()
df_binary['Sentiment'] = df_binary['Score'].apply(lambda x: 1 if x >= 4 else 0)

contractions_map = {
    "won't": "will not", "can't": "can not", "n't": " not",
    "'re": " are", "'s": "", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am"
}

def expand_contractions(text):
    text = text.lower()
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

stop_words = set(stopwords.words('english'))
negation_words = {'not', 'no', 'nor', 'never', 'none', 'nothing',
                   'neither', 'nowhere', 'cannot'}
stop_words = stop_words - negation_words

def remove_stopwords(text):
    words = text.split()
    return ' '.join(w for w in words if w not in stop_words)

def tag_negation(text, negation_words, window=3):
    words = text.split()
    result = []
    neg_counter = 0
    for w in words:
        if w == '!':
            neg_counter = 0
            result.append(w)
            continue
        if w in negation_words:
            neg_counter = window
            result.append(w)
        elif neg_counter > 0:
            result.append(w + '_NEG')
            neg_counter -= 1
        else:
            result.append(w)
    return ' '.join(result)


df_binary['clean_text'] = df_binary['Text'].apply(clean_text)
df_binary['final_text'] = df_binary['clean_text'].apply(remove_stopwords)
df_binary['final_text'] = df_binary['final_text'].apply(lambda t: tag_negation(t, negation_words))

X = df_binary['final_text']
y = df_binary['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

final_model = LinearSVC(C=0.1, class_weight='balanced', random_state=42, max_iter=5000)
final_model.fit(X_train_tfidf, y_train)

y_pred = final_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

import joblib

bundle = {
    'model': final_model,
    'tfidf': tfidf,
    'stop_words': stop_words
}
joblib.dump(bundle, 'ModelsOutcome/amazon_reviews_model.pkl')

print("model saved")