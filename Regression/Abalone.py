import  pandas as pd
df = pd.read_csv('DataSets/abalone.data.csv')
df.columns = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "WholeWeight",
    "ShuckedWeight",
    "VisceraWeight",
    "ShellWeight",
    "Rings"
]

#print (df.head())
#print(df.columns)
#print (df.describe())
#print(df.shape)
#print(df.isnull().sum())
x= df.drop("Rings", axis=1)
x["Sex"] =x["Sex"].map({"M":1, "F":2,"I":3})
y = df["Rings"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.4f}")

# import matplotlib.pyplot as plt
# plt.scatter(y_test, y_pred , alpha = 0.3)
# plt.xlabel("Actual Values") #Actual Rings
# plt.ylabel("Predicted Values") #Predicted Rings
# plt.title("Actual vs Predicted Values")
# plt.show()

import joblib

joblib.dump(regressor, 'ModelsOutcome/abalone_model.pkl')
joblib.dump(list(X_train.columns), 'ModelsOutcome/abalone_feature_columns.pkl')
print('model saved')




