import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_excel("DataSets/Real estate valuation data set.xlsx")
df.columns = [
    "TransactionDate",
    "HouseAge",
    "DistanceToMRT",
    "ConvenienceStores",
    "Latitude",
    "Longitude",
    "Price"
]

#print(df.head())
#print(df.columns)
#print (df.describe().to_string())
#print(df.shape)
#print(df.isnull().sum())
#print(df.info())
#print(df.corr().to_string())




#import missingno as sm
#sm.matrix(df)
#plt.show()
#sns.pairplot(df)
#plt.show()

#Find Skew------------------------
#plt.figure(figsize=(8,5))
#sns.histplot(df["Price"],kde=True)
#plt.title("Histogram of Real estate valuation data")
#plt.show()

# Find OutLayers-----------
#plt.figure(figsize=(10,6))
#sns.boxplot(data=df)
#plt.show()

#Heat Map------
#corr = df.corr()
#plt.figure(figsize=(10,8))
#sns.heatmap(corr, annot=True, cmap="YlGnBu")
#plt.show()

x = df.drop("Price", axis=1)
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

# print(x_train.shape)
# print(x_test.shape)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# print(y_pred)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# result = pd.DataFrame({
#     "Actual": y_test,
#     "Predicted": y_pred
# })
#
# print(result.head(20))

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title(
    "Actual vs Predicted"
)

plt.show()

