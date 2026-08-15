import pandas as pd
import numpy as np
from scipy.io import arff

data, meta = arff.loadarff('Datasets/Autism-Adult-Data.arff')
df = pd.DataFrame(data)
# print(df.shape)
# print(df.head())
# print(df.info())
# print(df.isnull().sum())

for col in df.select_dtypes(['object']).columns:
    df[col] = df[col].str.decode('utf-8')
df.replace('?', np.nan, inplace=True)
# print(df.isnull().sum())
# print(df.dtypes)
# print(df['gender'].unique())
# print(df['Class/ASD'].unique())
# print(df['age'].describe())

same_rows = (df['ethnicity'].isnull() == df['relation'].isnull()).sum()
# print({same_rows})
# print(df[df['age'] > 100])
# print(df[df['age'].isnull()])

df = df[df['age'] <= 100]
df = df.dropna(subset=['age'])
df['ethnicity'] = df['ethnicity'].fillna('Unknown')
# print(df['ethnicity'].value_counts())
df['relation'] = df['relation'].fillna('Unknown')
# print(df['relation'].value_counts())
# print(df.shape)
# print(df.isnull().sum().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# print(df['Class/ASD'].value_counts())
# print(df['Class/ASD'].value_counts(normalize=True) * 100)
#
# sns.countplot(data=df, x='Class/ASD')
# plt.title('Distribution of Class/ASD')
# plt.show()

# print(df[['age', 'result']].describe())
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# sns.histplot(df['age'], kde=True, ax=axes[0])
# axes[0].set_title('Age Distribution')
# sns.histplot(df['result'], kde=True, ax=axes[1])
# axes[1].set_title('Result Distribution')
# plt.tight_layout()
# plt.show()

# print(df.groupby('Class/ASD')['result'].describe())
# sns.boxplot(data=df, x='Class/ASD', y='result')
# plt.title('Result Score by Class/ASD')
# plt.show()
#
# a_cols = [f'A{i}_Score' for i in range(1, 11)]
# df_temp = df.copy()
# df_temp[a_cols] = df_temp[a_cols].astype(int)
# df_temp['target'] = (df_temp['Class/ASD'] == 'YES').astype(int)
# correlations = df_temp[a_cols + ['result', 'target']].corr()['target'].sort_values(ascending=False)
# print(correlations)
#
# for col in ['jundice', 'austim', 'gender', 'used_app_before']:
#     print(f"\n--- {col} ---")
#     print(pd.crosstab(df[col], df['Class/ASD'], normalize='index') * 100)
#
# sns.countplot(data=df, x='austim', hue='Class/ASD')
# plt.title('Family History of Autism vs Class/ASD')
# plt.show()

# print(df.groupby('Class/ASD')['age'].describe())
# sns.boxplot(data=df, x='Class/ASD', y='age')
# plt.title('Age by Class/ASD')
# plt.show()
#
# print(f"{df['contry_of_res'].nunique()}")
# print(df['contry_of_res'].value_counts())

df = df.drop(columns=['age_desc'])
a_cols = [f'A{i}_Score' for i in range(1, 11)]
df[a_cols] = df[a_cols].astype(int)
binary_map = {'no': 0, 'yes': 1}
df['jundice'] = df['jundice'].map(binary_map)
df['austim'] = df['austim'].map(binary_map)
df['used_app_before'] = df['used_app_before'].map(binary_map)
df['gender'] = df['gender'].map({'f': 0, 'm': 1})
# print(df.head())
# print(df.dtypes)

top_countries = df['contry_of_res'].value_counts().nlargest(5).index
# print( list(top_countries))
df['contry_of_res'] = df['contry_of_res'].apply(
    lambda x: x if x in top_countries else 'Other'
)
# print(df['contry_of_res'].value_counts())

# categorical_cols = ['ethnicity', 'contry_of_res', 'relation']
# df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# print(df_encoded.shape)
# print(df_encoded.columns.tolist())

df['ethnicity'] = df['ethnicity'].replace('others', 'Others')
# print(df['ethnicity'].value_counts())

categorical_cols = ['ethnicity', 'contry_of_res', 'relation']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# print(df_encoded.shape)

from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns=['Class/ASD'])
y = df_encoded['Class/ASD'].map({'NO': 0, 'YES': 1})
# print(X.shape, y.shape)
# print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# print(X_train.shape,X_test.shape)
# print(y_train.value_counts(normalize=True))
# print(y_test.value_counts(normalize=True))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
# print(X_train_scaled.describe().loc[['min', 'max']])

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    'Logistic Regression': (LogisticRegression(class_weight='balanced', random_state=42), X_train_scaled,
                            X_test_scaled),
    'KNN': (KNeighborsClassifier(n_neighbors=5), X_train_scaled, X_test_scaled),
    'Random Forest': (RandomForestClassifier(class_weight='balanced', random_state=42), X_train, X_test),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), X_train, X_test)
}
# results = []
# for name, (model, X_tr, X_te) in models.items():
#     model.fit(X_tr, y_train)
#     y_pred = model.predict(X_te)
#     results.append({
#         'Model': name,
#         'Accuracy': accuracy_score(y_test, y_pred),
#         'Precision': precision_score(y_test, y_pred),
#         'Recall': recall_score(y_test, y_pred),
#         'F1': f1_score(y_test, y_pred)
#     })
# results_df = pd.DataFrame(results)
# print(results_df)

from sklearn.model_selection import cross_val_score, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_results = []
# for name, (model, X_tr, X_te) in models.items():
#     scores = cross_val_score(model, X_tr, y_train, cv=skf, scoring='f1')
#     cv_results.append({
#         'Model': name,
#         'CV F1 Mean': scores.mean(),
#         'CV F1 Std': scores.std()
#     })
# cv_results_df = pd.DataFrame(cv_results)
# print(cv_results_df)

# final_model = RandomForestClassifier(class_weight='balanced', random_state=42)
# final_model.fit(X_train, y_train)
# importances = pd.Series(final_model.feature_importances_, index=X_train.columns)
# importances = importances.sort_values(ascending=False)
# print(importances.head(15))
# plt.figure(figsize=(10, 6))
# importances.head(15).plot(kind='barh')
# plt.gca().invert_yaxis()
# plt.title('Top 15 Feature Importances - Random Forest')
# plt.xlabel('Importance')
# plt.tight_layout()
# plt.show()
#Delete result feature
X_train_no_result = X_train.drop(columns=['result'])
X_test_no_result = X_test.drop(columns=['result'])
X_train_scaled_no_result = X_train_scaled.drop(columns=['result'])
X_test_scaled_no_result = X_test_scaled.drop(columns=['result'])
models_no_result = {
    'Logistic Regression': (LogisticRegression(class_weight='balanced', random_state=42), X_train_scaled_no_result, X_test_scaled_no_result),
    'KNN': (KNeighborsClassifier(n_neighbors=5), X_train_scaled_no_result, X_test_scaled_no_result),
    'Random Forest': (RandomForestClassifier(class_weight='balanced', random_state=42), X_train_no_result, X_test_no_result),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), X_train_no_result, X_test_no_result)
}
results_no_result = []
for name, (model, X_tr, X_te) in models_no_result.items():
    scores = cross_val_score(model, X_tr, y_train, cv=skf, scoring='f1')
    results_no_result.append({
        'Model': name,
        'CV F1 Mean': scores.mean(),
        'CV F1 Std': scores.std()
    })
results_no_result_df = pd.DataFrame(results_no_result)
# print(results_no_result_df)

final_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
final_model.fit(X_train_scaled_no_result, y_train)
y_pred_final = final_model.predict(X_test_scaled_no_result)
# print("Accuracy:", accuracy_score(y_test, y_pred_final))
# print("Precision:", precision_score(y_test, y_pred_final))
# print("Recall:", recall_score(y_test, y_pred_final))
# print("F1:", f1_score(y_test, y_pred_final))
coefficients = pd.Series(final_model.coef_[0], index=X_train_scaled_no_result.columns)
coefficients = coefficients.sort_values(ascending=False)
# print("\nتوپ ۱۰ ویژگی با بیشترین تاثیر مثبت:\n", coefficients.head(10))
# print("\nتوپ ۵ ویژگی با بیشترین تاثیر منفی:\n", coefficients.tail(5))

# import joblib
#
# joblib.dump(final_model, 'ModelsOutcome/autism_lr_model.pkl')
# joblib.dump(list(X_train_scaled_no_result.columns), 'ModelsOutcome/autism_model_features.pkl')
# joblib.dump(scaler, 'ModelsOutcome/autism_scaler.pkl')
#
# print("model saved")
#
