import pandas as pd
train = pd.read_csv('Datasets/RoomOccupancy/datatraining.txt', sep=',')
test1 = pd.read_csv('Datasets/RoomOccupancy/datatest.txt', sep=',')
test2 = pd.read_csv('Datasets/RoomOccupancy/datatest2.txt', sep=',')
# for name, df in [('Train', train), ('Test1', test1), ('Test2', test2)]:
#     print(f"\n{'='*20} {name} {'='*20}")
#     print(df.info())
#     print(df.head())
#
for name, df in [('Train', train), ('Test1', test1), ('Test2', test2)]:
    df['date'] = pd.to_datetime(df['date'])
#     print(f"\n{'='*20} {name} {'='*20}")
#     print(f"بازه زمانی: {df['date'].min()}  تا  {df['date'].max()}")
#     print(f"توزیع Occupancy:\n{df['Occupancy'].value_counts(normalize=True)}")
# # Test1  →  [gap ~7h]  →  Train  →  [gap ~29h]  →  Test2

# for name, df in [('Train', train), ('Test1', test1), ('Test2', test2)]:
#     diffs = df['date'].diff().dropna()
#     print(f"\n{'='*20} {name} {'='*20}")
#     print(diffs.value_counts().head(5))
#     print(f"{diffs.max()}")

import matplotlib.pyplot as plt
import seaborn as sns

# features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()
# for i, feat in enumerate(features):
#     sns.boxplot(data=train, x='Occupancy', y=feat, ax=axes[i])
#     axes[i].set_title(feat)
# fig.delaxes(axes[5])
# plt.tight_layout()
# plt.show()
# print(train.groupby('Occupancy')[features].describe().T)

# plt.figure(figsize=(8,6))
# sns.heatmap(train[features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix - Train')
# plt.show()
# print(train[features].corr())

train = train.drop(columns=['Humidity'])
test1 = test1.drop(columns=['Humidity'])
test2 = test2.drop(columns=['Humidity'])
# print(train.columns.tolist())

feature_cols = ['Temperature', 'Light', 'CO2', 'HumidityRatio']
X_train = train[feature_cols]
y_train = train['Occupancy']

X_test1 = test1[feature_cols]
y_test1 = test1['Occupancy']

X_test2 = test2[feature_cols]
y_test2 = test2['Occupancy']
# print(X_train.shape, X_test1.shape, X_test2.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test1)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test1, y_pred),
        'Precision': precision_score(y_test1, y_pred),
        'Recall': recall_score(y_test1, y_pred),
        'F1': f1_score(y_test1, y_pred)
    })
results_df = pd.DataFrame(results)
# print(results_df)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
best_model = models['Logistic Regression']
# y_pred_lr = best_model.predict(X_test1)
# cm = confusion_matrix(y_test1, y_pred_lr)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Empty (0)', 'Occupied (1)'])
# disp.plot(cmap='Blues')
# plt.title('Confusion Matrix - Logistic Regression on Test1')
# plt.show()
# print(cm)

y_pred_test2 = best_model.predict(X_test2)

# print(f"Accuracy:  {accuracy_score(y_test2, y_pred_test2):.4f}")
# print(f"Precision: {precision_score(y_test2, y_pred_test2):.4f}")
# print(f"Recall:    {recall_score(y_test2, y_pred_test2):.4f}")
# print(f"F1:        {f1_score(y_test2, y_pred_test2):.4f}")
#
# cm2 = confusion_matrix(y_test2, y_pred_test2)
# print(cm2)
#
# disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=['Empty (0)', 'Occupied (1)'])
# disp2.plot(cmap='Greens')
# plt.title('Confusion Matrix - Logistic Regression on Test2 (Final)')
# plt.show()

import joblib

joblib.dump(best_model, 'ModelsOutcome/room_occupancy_logreg_model.pkl')
joblib.dump(feature_cols, 'ModelsOutcome/room_occupancy_feature_columns.pkl')
print("مدل و اسم فیچرها ذخیره شدن.")
# print(feature_cols)

loaded_model = joblib.load('ModelsOutcome/room_occupancy_logreg_model.pkl')
loaded_features = joblib.load('ModelsOutcome/room_occupancy_feature_columns.pkl')
