import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression



cols = ['mpg','cylinders','displacement','horsepower',
        'weight','acceleration','model_year','origin','car_name']

df = pd.read_csv('DataSets/auto-mpg.data', sep=r'\s+', names=cols, na_values='?')
#print(df.head())
#print(df.isnull().sum())

# df['horsepower'].hist(bins=30)
# plt.title('horsepower distribution')
# plt.show()
# print(df['horsepower'].describe())
# #It's not normal. so we use median for null values.

df.drop(columns=['car_name'], inplace=True)
df['horsepower']= df['horsepower'].fillna(df['horsepower'].median())
sns.heatmap(df.corr(), annot=True,cmap='coolwarm')

df = pd.get_dummies(df, columns=['origin'], drop_first=True)

x = df.drop('mpg', axis=1)
y = df['mpg']

x_train ,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train_s, y_train)

y_pred = model.predict(x_test_s)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_score = r2_score(y_test, y_pred)
#print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
print(f"R²  : {r2_score:.3f}")
print(f"MSE : {mse:.3f}")
print(f"RMSE: {rmse:.3f} mpg")
print(f"MAE : {mae:.3f} mpg")

joblib.dump(model,'mpg_model.pkl')
joblib.dump(scaler,'scaler_model.pkl')
print('Model saved!')
