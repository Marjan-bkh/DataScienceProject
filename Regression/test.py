import pandas as pd
df = pd.read_csv('DataSets/forecast_pullution.csv')
# print(df['cbwd'])
df = pd.get_dummies(df, columns=["cbwd"], prefix="wd")
print(df.columns)
wind_cols = [c for c in df.columns if c.startswith("wd_")]

print(wind_cols)