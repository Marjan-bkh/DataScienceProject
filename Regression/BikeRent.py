import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# تنظیمات نمایش بهتر
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

df_day = pd.read_csv('DataSets/BikeRent_hour.csv')
df_hour = pd.read_csv('DataSets/BikeRent_day.csv')

# print("=== day.csv ===")
# print(df_day.shape)
# print(df_day.info())
# print(df_day.head())
#
# print("\n=== hour.csv ===")
# print(df_hour.shape)
# print(df_hour.info())
# print(df_hour.head())

print(df_day.describe())
print(df_hour.describe())