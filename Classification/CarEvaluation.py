import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# چون فایل car.data هدر (اسم ستون) نداره، خودمون اسم ستون‌ها رو مشخص می‌کنیم
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

df = pd.read_csv('Datasets/car.data', names=column_names)
# print(df.shape)
# print(df.head())
# print(df.info())
# # بررسی مقادیر یکتا در هر ستون
# for col in df.columns:
#     print(f"--- {col} ---")
#     print(df[col].value_counts())
#     print()
# plt.figure(figsize=(8, 5))
# sns.countplot(data=df, x='class', order=df['class'].value_counts().index, palette='viridis')
# plt.title('توزیع کلاس هدف (Class Distribution)')
# plt.xlabel('کلاس')
# plt.ylabel('تعداد نمونه')
#
# # نمایش درصد روی هر ستون
# total = len(df)
# for p in plt.gca().patches:
#     percentage = f'{100 * p.get_height() / total:.1f}%'
#     plt.gca().annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
#                         ha='center', va='bottom')
# plt.tight_layout()
# plt.show()

features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
#
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# axes = axes.flatten()
#
# for i, col in enumerate(features):
#     # جدول متقاطع: برای هر مقدار ستون، چند تا از هر کلاس داریم
#     cross_tab = pd.crosstab(df[col], df['class'])
#     cross_tab.plot(kind='bar', stacked=True, ax=axes[i], colormap='viridis')
#     axes[i].set_title(f'{col} بر اساس class')
#     axes[i].set_xlabel(col)
#     axes[i].set_ylabel('تعداد')
#     axes[i].legend(title='class', fontsize=8)
#     axes[i].tick_params(axis='x', rotation=0)
#
# plt.tight_layout()
# plt.show()

from scipy.stats import chi2_contingency

def cramers_v(x, y):
    """محاسبه‌ی Cramér's V بین دو متغیر categorical"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

print("همبستگی هر ویژگی با class (Cramér's V):\n")
for col in features:
    v = cramers_v(df[col], df['class'])
    print(f"{col}: {v:.3f}")