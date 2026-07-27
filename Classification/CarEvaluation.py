import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# چون فایل car.data هدر (اسم ستون) نداره، خودمون اسم ستون‌ها رو مشخص می‌کنیم
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

df = pd.read_csv('Datasets/car.data', names=column_names)

print(df.shape)
print(df.head())
print(df.info())