import pandas as pd
import numpy as np

df = pd.read_csv(
    'Datasets/household_power_consumption.txt',
    sep=';',
    na_values=['?'],
    low_memory=False
)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.drop(columns=['Date', 'Time'], inplace=True)
# print(df.shape)
# print(df.dtypes)
# print(df.head())
# print(df.isnull().sum())
# print(df.memory_usage(deep=True).sum() / 1024**2, "MB")

# missing_mask = df['Global_active_power'].isnull()
# # print(df.loc[missing_mask, 'Datetime'].min())
# # print(df.loc[missing_mask, 'Datetime'].max())
#
# missing_dates = df.loc[missing_mask, 'Datetime'].dt.date
# # print(missing_dates.value_counts().sort_index().head(20))
# # print("تعداد روزهای متفاوتی که حداقل یک missing دارن:", missing_dates.nunique())
# missing_by_day = missing_dates.value_counts().sort_values(ascending=False)
# print(missing_by_day.head(15))
# print(missing_by_day[missing_by_day > 1000])

df = df.sort_values('Datetime').reset_index(drop=True)
cols_to_fill = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
df[cols_to_fill] = df[cols_to_fill].interpolate(method='linear', limit=60, limit_direction='forward')
# print(df[cols_to_fill].isnull().sum())

df = df.set_index('Datetime')
daily = df.resample('D').agg({
    'Global_active_power': ['mean', 'max', 'std', 'sum'],
    'Global_reactive_power': 'mean',
    'Voltage': 'mean',
    'Global_intensity': 'mean',
    'Sub_metering_1': 'sum',
    'Sub_metering_2': 'sum',
    'Sub_metering_3': 'sum'
})
daily.columns = ['_'.join(col) for col in daily.columns]
missing_ratio = df['Global_active_power'].isnull().resample('D').mean()
daily['missing_ratio'] = missing_ratio
# print(daily.shape)
# print(daily.head())
# print(daily['missing_ratio'].describe())

threshold = 0.1  # اگه بیش از 10% دقایق یک روز گمشده باشه، اون روز حذف میشه
daily_clean = daily[daily['missing_ratio'] <= threshold].copy()
daily_clean = daily_clean.drop(columns=['missing_ratio'])

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(15, 4))
daily_clean['Global_active_power_mean'].plot(ax=ax)
ax.set_title('روند میانگین مصرف روزانه در طول زمان')
plt.tight_layout()
plt.show()

daily_clean.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()

corr = daily_clean.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.tight_layout()
plt.show()

print(corr['Global_active_power_mean'].sort_values(ascending=False))
