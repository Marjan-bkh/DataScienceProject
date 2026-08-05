import pandas as pd
import numpy as np

df = pd.read_csv('Datasets/time-series-19-covid-combined.csv')
# print(df.shape)
# print(df.columns.tolist())
# print(df.head(10))
# print(df.dtypes)
# print(df.isnull().sum())

df['Date'] = pd.to_datetime(df['Date'])
df_country = df.groupby(['Country/Region', 'Date'], as_index=False)[['Confirmed', 'Recovered', 'Deaths']].sum()
# print(df_country.shape)
# print(df_country['Country/Region'].nunique())
# print(df_country['Date'].min(), '-', df_country['Date'].max())
# print(df_country.isnull().sum())
# print(df_country.head(10))

df_country = df_country.sort_values(['Country/Region', 'Date'])
df_country['New_Confirmed'] = df_country.groupby('Country/Region')['Confirmed'].diff().fillna(0)
df_country['New_Deaths'] = df_country.groupby('Country/Region')['Deaths'].diff().fillna(0)
df_country['New_Recovered'] = df_country.groupby('Country/Region')['Recovered'].diff().fillna(0)
# print(df_country[df_country['Country/Region']=='Afghanistan'].head(15))
# print((df_country[['New_Confirmed','New_Deaths','New_Recovered']] < 0).sum())

df_country['New_Confirmed'] = df_country['New_Confirmed'].clip(lower=0)
df_country['New_Deaths'] = df_country['New_Deaths'].clip(lower=0)
df_country['New_Recovered'] = df_country['New_Recovered'].clip(lower=0)
# print((df_country[['New_Confirmed','New_Deaths','New_Recovered']] < 0).sum())

df_country = df_country.set_index('Date')
weekly = (
    df_country
    .groupby('Country/Region')
    .resample('W')
    .agg({
        'New_Confirmed': 'sum',
        'New_Deaths': 'sum',
        'New_Recovered': 'sum',
        'Confirmed': 'last',
        'Deaths': 'last',
        'Recovered': 'last'
    })
    .reset_index()
)
# print(weekly.shape)
# print(weekly['Country/Region'].nunique())
# print(weekly.head(15))

weekly = weekly.sort_values(['Country/Region', 'Date'])
weekly['Prev_New_Confirmed'] = weekly.groupby('Country/Region')['New_Confirmed'].shift(1)
weekly['Growth_Rate'] = (weekly['New_Confirmed'] - weekly['Prev_New_Confirmed']) / weekly['Prev_New_Confirmed'].replace(0, np.nan)
weekly['CFR'] = weekly['Deaths'] / weekly['Confirmed'].replace(0, np.nan) #Case Fatality Rate
# print(weekly[['Growth_Rate', 'CFR']].describe())
# print(weekly['Growth_Rate'].isnull().sum())
# print(weekly['CFR'].isnull().sum())

q_low = weekly['Growth_Rate'].quantile(0.01)
q_high = weekly['Growth_Rate'].quantile(0.99)
# print(q_low, q_high)

weekly['Growth_Rate'] = weekly['Growth_Rate'].clip(lower=q_low, upper=q_high)
# print(weekly['Growth_Rate'].describe())

# print(weekly[weekly['Growth_Rate'].isnull()]['Confirmed'].describe())
# print(weekly[weekly['CFR'].isnull()]['Confirmed'].describe())

weekly = weekly.dropna(subset=['CFR']).copy()
weekly['Growth_Rate'] = weekly['Growth_Rate'].fillna(0)
# print(weekly.shape)
# print(weekly.isnull().sum())

weekly['Growth_Rate_norm'] = (weekly['Growth_Rate'] - weekly['Growth_Rate'].min()) / (weekly['Growth_Rate'].max() - weekly['Growth_Rate'].min())
weekly['CFR_norm'] = (weekly['CFR'] - weekly['CFR'].min()) / (weekly['CFR'].max() - weekly['CFR'].min())
weekly['Risk_Score'] = 0.5 * weekly['Growth_Rate_norm'] + 0.5 * weekly['CFR_norm']
# print(weekly['Risk_Score'].describe())

weekly['Risk_Level'] = pd.qcut(weekly['Risk_Score'], q=3, labels=['Low', 'Medium', 'High'])
# print(weekly['Risk_Level'].value_counts())
# print(weekly.groupby('Risk_Level')['Risk_Score'].describe())

feature_cols = ['New_Confirmed', 'New_Deaths', 'New_Recovered', 'Confirmed', 'Deaths', 'Recovered']
X = weekly[feature_cols]
y = weekly['Risk_Level']
# print(X.shape, y.shape)
# print(X.describe())

for col in feature_cols:
    X[col + '_log'] = np.log1p(X[col])
log_cols = [c + '_log' for c in feature_cols]
# print(X[log_cols].describe())

X_final = X[log_cols].copy()
X_final.columns = feature_cols
# print(X_final.head())
# print(X_final.shape)

weekly = weekly.sort_values('Date')
# print(weekly['Date'].min(), weekly['Date'].max())
cutoff_date = weekly['Date'].quantile(0.8, interpolation='nearest')
# print(cutoff_date)
train_mask = weekly['Date'] <= cutoff_date
test_mask = weekly['Date'] > cutoff_date
# print(train_mask.sum(), test_mask.sum())

# print((X_final.index == weekly.index).all())
X_train = X_final.loc[weekly.index[train_mask]]
X_test = X_final.loc[weekly.index[test_mask]]
y_train = y.loc[weekly.index[train_mask]]
y_test = y.loc[weekly.index[test_mask]]
# print(X_train.shape, X_test.shape)
# print(y_train.value_counts(normalize=True))
# print(y_test.value_counts(normalize=True))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# models = {
#     'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=1000), X_train_scaled, X_test_scaled),
#     'Random Forest': (RandomForestClassifier(class_weight='balanced', random_state=42), X_train, X_test),
#     'Gradient Boosting': (GradientBoostingClassifier(random_state=42), X_train, X_test),
# }
# results = []
# for name, (model, X_tr, X_te) in models.items():
#     model.fit(X_tr, y_train)
#     y_pred = model.predict(X_te)
#     results.append({
#         'Model': name,
#         'Accuracy': accuracy_score(y_test, y_pred),
#         'Precision': precision_score(y_test, y_pred, average='macro'),
#         'Recall': recall_score(y_test, y_pred, average='macro'),
#         'F1': f1_score(y_test, y_pred, average='macro')
#     })
# results_df = pd.DataFrame(results)
# # print(results_df)

from sklearn.metrics import confusion_matrix, classification_report

best_model = RandomForestClassifier(class_weight='balanced', random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
# print(confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High']))
# print(classification_report(y_test, y_pred, labels=['Low', 'Medium', 'High']))
importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
# print(importances)

weekly = weekly.sort_values(['Country/Region', 'Date'])
weekly['Growth_Rate_lag1'] = weekly.groupby('Country/Region')['Growth_Rate'].shift(1)
weekly['Growth_Rate_lag2'] = weekly.groupby('Country/Region')['Growth_Rate'].shift(2)
weekly['CFR_lag1'] = weekly.groupby('Country/Region')['CFR'].shift(1)
weekly['CFR_lag2'] = weekly.groupby('Country/Region')['CFR'].shift(2)
weekly['Growth_Rate_rolling3'] = weekly.groupby('Country/Region')['Growth_Rate'].transform(lambda x: x.shift(1).rolling(3).mean())
# print(weekly[['Growth_Rate_lag1','Growth_Rate_lag2','CFR_lag1','CFR_lag2','Growth_Rate_rolling3']].isnull().sum())

weekly_clean = weekly.dropna(subset=['Growth_Rate_lag1','Growth_Rate_lag2','CFR_lag1','CFR_lag2','Growth_Rate_rolling3']).copy()
# print(weekly_clean.shape)
feature_cols_v2 = ['Growth_Rate_lag1', 'Growth_Rate_lag2', 'CFR_lag1', 'CFR_lag2', 'Growth_Rate_rolling3']
X2 = weekly_clean[feature_cols_v2]
y2 = weekly_clean['Risk_Level']
# print(X2.describe())

weekly_clean = weekly_clean.sort_values('Date')
cutoff_date2 = weekly_clean['Date'].quantile(0.8, interpolation='nearest')
# print(cutoff_date2)
train_mask2 = weekly_clean['Date'] <= cutoff_date2
test_mask2 = weekly_clean['Date'] > cutoff_date2
X2_train = X2.loc[weekly_clean.index[train_mask2]]
X2_test = X2.loc[weekly_clean.index[test_mask2]]
y2_train = y2.loc[weekly_clean.index[train_mask2]]
y2_test = y2.loc[weekly_clean.index[test_mask2]]
# print(X2_train.shape, X2_test.shape)
# print(y2_train.value_counts(normalize=True))
# print(y2_test.value_counts(normalize=True))

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)
models2 = {
    'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=1000), X2_train_scaled, X2_test_scaled),
    'Random Forest': (RandomForestClassifier(class_weight='balanced', random_state=42), X2_train, X2_test),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), X2_train, X2_test),
}
results2 = []
for name, (model, X_tr, X_te) in models2.items():
    model.fit(X_tr, y2_train)
    y_pred = model.predict(X_te)
    results2.append({
        'Model': name,
        'Accuracy': accuracy_score(y2_test, y_pred),
        'Precision': precision_score(y2_test, y_pred, average='macro'),
        'Recall': recall_score(y2_test, y_pred, average='macro'),
        'F1': f1_score(y2_test, y_pred, average='macro')
    })
results_df2 = pd.DataFrame(results2)
# print(results_df2)

best_model2 = GradientBoostingClassifier(random_state=42)
best_model2.fit(X2_train, y2_train)
y_pred2 = best_model2.predict(X2_test)

# print(confusion_matrix(y2_test, y_pred2, labels=['Low', 'Medium', 'High']))
# print(classification_report(y2_test, y_pred2, labels=['Low', 'Medium', 'High']))
# importances2 = pd.Series(best_model2.feature_importances_, index=X2_train.columns).sort_values(ascending=False)
# print(importances2)

import joblib

joblib.dump(best_model2, 'ModelsOutcome/covid_risk_gb_model.pkl')
joblib.dump(list(X2_train.columns), 'ModelsOutcome/covid_risk_features.pkl')

print("Saved successfully")