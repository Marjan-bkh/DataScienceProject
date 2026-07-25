import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_excel('DataSets/Concrete_Data.xls')

# نام ستون‌ها رو ساده می‌کنیم چون اسم‌های اصلی طولانی و شلوغن
df.columns = ['cement', 'slag', 'flyash', 'water',
              'superplasticizer', 'coarse_agg', 'fine_agg',
              'age', 'strength']

# print(df.head())
# print(df.describe())
# print(df.info)
# print(df.isnull().sum())
# print(df.columns)
# print(df.shape)

# df.hist(figsize=(14, 10), bins=30, edgecolor='black')
# plt.tight_layout()
# # plt.savefig('histograms.png', dpi=100)
# plt.show()

# plt.figure(figsize=(10, 8))
# corr_matrix = df.corr()
# sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
#             center=0, square=True, linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.tight_layout()
# # plt.savefig('heatmap.png', dpi=100)
# plt.show()

# features = ['cement', 'slag', 'flyash', 'water',
#             'superplasticizer', 'coarse_agg', 'fine_agg', 'age']
#
# fig, axes = plt.subplots(2, 4, figsize=(18, 8))
# axes = axes.flatten()
#
# for i, col in enumerate(features):
#     axes[i].scatter(df[col], df['strength'], alpha=0.4, s=15)
#     axes[i].set_xlabel(col)
#     axes[i].set_ylabel('strength')
#     axes[i].set_title(f'{col} vs strength')
#
# plt.tight_layout()
# # plt.savefig('scatter_all.png', dpi=100)
# plt.show()

print('درصد نمونه‌های بدون slag:', (df['slag']==0).mean()*100, '%')
print('درصد نمونه‌های بدون flyash:', (df['flyash']==0).mean()*100, '%')

from sklearn.model_selection import train_test_split
X = df.drop('strength', axis=1)
Y= df['strength']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
print('Train Shape:', X_train.shape)
print('Test Shape:', X_test.shape)

def add_features(X):
    X = X.copy()
    # نسبت آب به سیمان — مهم‌ترین فاکتور شناخته‌شده در مهندسی بتن
    X['water_cement_ratio'] = X['water'] / X['cement']
    # مجموع مواد چسباننده (سیمان + جایگزین‌ها)
    X['total_cementitious'] = X['cement'] + X['slag'] + X['flyash']
    # نسبت فوق‌روان‌کننده به مواد چسباننده
    X['superplasticizer_ratio'] = X['superplasticizer'] / X['total_cementitious']
    # نسبت سنگدونه ریز به درشت
    X['agg_ratio'] = X['fine_agg'] / X['coarse_agg']
    # لگاریتم سن (چون توزیعش خیلی چوله بود)
    X['log_age'] = np.log1p(X['age'])
    return X

X_train_fe = add_features(X_train)
X_test_fe = add_features(X_test)

print(X_train_fe.columns.tolist())
print(X_train_fe.head())

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

lr=LinearRegression()
lr.fit(X_train_fe, Y_train)
Y_pred_lr = lr.predict(X_test_fe)

rmse_lr= np.sqrt(mean_squared_error(Y_test, Y_pred_lr))
mae_lr = mean_absolute_error(Y_test, Y_pred_lr)
r2_lr = r2_score(Y_test, Y_pred_lr)
print(f'Linear Regression -> RMSE: {rmse_lr:.3f}, MAE: {mae_lr:.3f}, R2: {r2_lr:.3f}')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# --- Random Forest ---
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_fe, Y_train)
Y_pred_rf = rf.predict(X_test_fe)

rmse_rf = np.sqrt(mean_squared_error(Y_test, Y_pred_rf))
mae_rf = mean_absolute_error(Y_test, Y_pred_rf)
r2_rf = r2_score(Y_test, Y_pred_rf)
print(f'Random Forest -> RMSE: {rmse_rf:.3f}, MAE: {mae_rf:.3f}, R2: {r2_rf:.3f}')

# --- Gradient Boosting ---
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                 max_depth=3, random_state=42)
gb.fit(X_train_fe, Y_train)
Y_pred_gb = gb.predict(X_test_fe)

rmse_gb = np.sqrt(mean_squared_error(Y_test, Y_pred_gb))
mae_gb = mean_absolute_error(Y_test, Y_pred_gb)
r2_gb = r2_score(Y_test, Y_pred_gb)
print(f'Gradient Boosting -> RMSE: {rmse_gb:.3f}, MAE: {mae_gb:.3f}, R2: {r2_gb:.3f}')

from sklearn.model_selection import cross_val_score,KFold
# چون Feature Engineering ما فقط row-wise هست (نه آماری)، می‌تونیم راحت
# روی کل X (بعد از اعمال add_features) از CV استفاده کنیم
X_all_fe =add_features(X)
kf= KFold(n_splits=5, shuffle=True, random_state=42)
models = {
    'Linear Regression': lr,
    'Random Forest': rf,
    'Gradient Boosting': gb
}
for name, model in models.items():
    scores = cross_val_score(model,X_all_fe,Y, cv=kf,scoring='r2',n_jobs=-1)
    print(f'{name}: R2 mean={scores.mean():.3f}, std={scores.std():.3f}')
    print(f'   individual folds: {np.round(scores, 3)}')

# Feature Importance از Gradient Boosting (بهترین مدل ما)
importance_gb= pd.Series(gb.feature_importances_,index=X_train_fe.columns)
importance_gb= importance_gb.sort_values(ascending=False)
print(importance_gb)

plt.figure(figsize=(10, 6))
importance_gb.plot(kind='barh')
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.title('Feature Importance - Gradient Boosting')
plt.tight_layout()
# plt.savefig('feature_importance.png', dpi=100)
# plt.show()
import joblib

# نسخه نهایی فیچرها - حذف فیچرهای کم‌اهمیت و redundant
def add_features_final(X):
    X = X.copy()
    X['water_cement_ratio'] = X['water'] / X['cement']
    X['total_cementitious'] = X['cement'] + X['slag'] + X['flyash']
    X['agg_ratio'] = X['fine_agg'] / X['coarse_agg']
    X['log_age'] = np.log1p(X['age'])
    X = X.drop(columns=['age'])  # چون log_age جایگزینش شده
    return X

X_final = add_features_final(X)

# مدل نهایی روی کل داده (train + test) train می‌شه چون آماده‌ی استفاده واقعیه
final_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                          max_depth=3, random_state=42)
final_model.fit(X_final, Y)
#
# # ذخیره مدل برای استفاده بعدی
# joblib.dump(final_model, 'concrete_strength_model.pkl')
# print('مدل ذخیره شد.')

# # تست یک پیش‌بینی نمونه
# sample = X_final.iloc[[0]]
# pred = final_model.predict(sample)
# print(f'پیش‌بینی نمونه: {pred[0]:.2f} MPa | مقدار واقعی: {Y.iloc[0]:.2f} MPa')