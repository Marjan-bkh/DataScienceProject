import pandas as pd

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

df = pd.read_csv('Datasets/PersonsIncome/adult.data', names=columns,engine='python', sep=r',\s*', na_values='?')
# print(df.shape)
# print(df.head())
# print(df.info())
# print(df['income'].value_counts())
# print(df['income'].value_counts(normalize=True)*100)
# print(df.describe())
categorical_features =['workclass', 'education', 'marital-status', 'occupation', 'relationship'
                        , 'race', 'sex','native-country']
# for col in categorical_features:
#     percent= df[col].value_counts(normalize=True)*100
#     print(percent.head(10))

threshold = 1.0  # %
for col in categorical_features:
    percent = df[col].value_counts(normalize=True) * 100
    rare_categories = percent[percent < threshold].index
    df[col] = df[col].replace(rare_categories, 'Other')
    # print(col, '->', list(rare_categories))

df['workclass'] = df['workclass'].fillna('Unknown')
df['occupation'] = df['occupation'].fillna('Unknown')
df['native-country'] = df['native-country'].fillna('Unknown')
# print(df.isnull().sum())
df = df.drop('fnlwgt', axis=1)
# print(df.groupby('education')['education-num'].unique())
df = df.drop('education', axis=1)
# print(df.shape)
# print((df['capital-gain'] == 0).sum() / len(df) * 100)
# print((df['capital-loss'] == 0).sum() / len(df) * 100)
# print(df[df['capital-gain'] > 0]['capital-gain'].describe())
# print(df[df['capital-loss'] > 0]['capital-loss'].describe())
# print((df['capital-gain'] == 99999).sum())

# Target encoding
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
# Binary encoding for sex
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
# One-Hot Encoding for nominal features
nominal_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
df = pd.get_dummies(df, columns=nominal_features, drop_first=True)
# print(df.shape)

test_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

df_test = pd.read_csv('Datasets/PersonsIncome/adult.test',names=test_columns,
                      sep=r',\s*', engine='python',na_values='?', skiprows=1)
df_test['income'] = df_test['income'].str.rstrip('.')

# لیست دسته‌های نادر که قبلاً از train به‌دست اومد (بر اساس آستانه ۱٪ روی train)
rare_map = {
    'workclass': ['Without-pay', 'Never-worked'],
    'education': ['1st-4th', 'Preschool'],
    'marital-status': ['Married-AF-spouse'],
    'occupation': ['Priv-house-serv', 'Armed-Forces'],
    'race': ['Amer-Indian-Eskimo', 'Other'],
    'native-country': ['Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador',
                        'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy',
                        'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland',
                        'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua',
                        'Peru', 'France', 'Greece', 'Ecuador', 'Ireland', 'Hong',
                        'Cambodia', 'Trinadad&Tobago', 'Thailand', 'Laos', 'Yugoslavia',
                        'Outlying-US(Guam-USVI-etc)', 'Honduras', 'Hungary', 'Scotland',
                        'Holand-Netherlands']
}
for col, rare_categories in rare_map.items():
    df_test[col] = df_test[col].replace(rare_categories, 'Other')

df_test['workclass'] = df_test['workclass'].fillna('Unknown')
df_test['occupation'] = df_test['occupation'].fillna('Unknown')
df_test['native-country'] = df_test['native-country'].fillna('Unknown')
df_test = df_test.drop(['fnlwgt', 'education'], axis=1)
df_test['income'] = df_test['income'].map({'<=50K': 0, '>50K': 1})
df_test['sex'] = df_test['sex'].map({'Male': 0, 'Female': 1})

# One-Hot Encoding
nominal_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
df_test = pd.get_dummies(df_test, columns=nominal_features, drop_first=True)

X_train_final = df.drop('income', axis=1)
y_train_final = df['income']
X_test_final = df_test.drop('income', axis=1)
y_test_final = df_test['income']
X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)
# print(X_train_final.shape, X_test_final.shape)
# print(y_test_final.value_counts(normalize=True))
# print((X_train_final.columns == X_test_final.columns).all())

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)
lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
knn_model = KNeighborsClassifier()
models = {
    'Logistic Regression': (lr_model, X_train_scaled, X_test_scaled),
    'Random Forest': (rf_model, X_train_final, X_test_final),
    'KNN': (knn_model, X_train_scaled, X_test_scaled)
}
results = []
for name, (model, X_tr, X_te) in models.items():
    model.fit(X_tr, y_train_final)
    y_pred = model.predict(X_te)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test_final, y_pred),
        'Precision': precision_score(y_test_final, y_pred),
        'Recall': recall_score(y_test_final, y_pred),
        'F1': f1_score(y_test_final, y_pred)
    })
results_df = pd.DataFrame(results)
# print(results_df)

# from sklearn.model_selection import StratifiedKFold, cross_val_score
#
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_results = []
# for name, (model, X_tr, X_te) in models.items():
#     scores = cross_val_score(model, X_tr, y_train_final, cv=cv, scoring='f1')
#     cv_results.append({
#         'Model': name,
#         'F1 Mean': scores.mean(),
#         'F1 Std': scores.std()
#     })
# cv_results_df = pd.DataFrame(cv_results)
# # print(cv_results_df)

import joblib

joblib.dump(rf_model, 'ModelsOutcome/adult_income_rf_model.pkl')
joblib.dump(list(X_train_final.columns), 'ModelsOutcome/adult_income_columns.pkl')
print("مدل و اسم فیچرها ذخیره شدن.")
#
# columns = joblib.load('adult_income_columns.pkl')
# X_new = X_new.reindex(columns=columns, fill_value=0)