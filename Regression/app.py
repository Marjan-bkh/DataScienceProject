import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

# بارگذاری مدل و scaler
model  = joblib.load('mpg_model.pkl')
scaler = joblib.load('mpg_scaler.pkl')

# ساخت پنجره اصلی
root = tk.Tk()
root.title('پیش‌بینی مصرف سوخت خودرو')
root.geometry('400x500')
root.resizable(False, False)

# تابع پیش‌بینی
def predict():
    try:
        origin_val = origin_var.get()
        origin_2   = 1 if origin_val == 'اروپا' else 0
        origin_3   = 1 if origin_val == 'ژاپن'  else 0

        new_car = pd.DataFrame([{
            'cylinders'   : int(cylinders_var.get()),
            'displacement': float(displacement_var.get()),
            'horsepower'  : float(horsepower_var.get()),
            'weight'      : float(weight_var.get()),
            'acceleration': float(acceleration_var.get()),
            'model_year'  : int(modelyear_var.get()),
            'origin_2'    : origin_2,
            'origin_3'    : origin_3
        }])

        scaled = scaler.transform(new_car)
        result = model.predict(scaled)[0]

        result_label.config(
            text=f'مصرف سوخت پیش‌بینی شده:\n{result:.1f} mpg',
            foreground='green'
        )

    except ValueError:
        messagebox.showerror('خطا', 'لطفاً همه فیلدها را پر کنید!')

# ساخت فیلدها
fields = [
    ('تعداد سیلندر (3-8)',       'cylinders_var',    '4'),
    ('حجم موتور (68-455)',       'displacement_var', '120'),
    ('قدرت موتور (46-230)',      'horsepower_var',   '90'),
    ('وزن خودرو (1600-5200)',    'weight_var',       '2500'),
    ('شتاب (8-25)',              'acceleration_var', '15'),
    ('سال تولید (70-82)',        'modelyear_var',    '80'),
]

# دیکشنری برای نگه داشتن متغیرها
vars_dict = {}

for i, (label_text, var_name, default) in enumerate(fields):
    ttk.Label(root, text=label_text).grid(
        row=i, column=0, padx=20, pady=8, sticky='w'
    )
    var = tk.StringVar(value=default)
    vars_dict[var_name] = var
    ttk.Entry(root, textvariable=var, width=15).grid(
        row=i, column=1, padx=20, pady=8
    )

# اتصال متغیرها
cylinders_var    = vars_dict['cylinders_var']
displacement_var = vars_dict['displacement_var']
horsepower_var   = vars_dict['horsepower_var']
weight_var       = vars_dict['weight_var']
acceleration_var = vars_dict['acceleration_var']
modelyear_var    = vars_dict['modelyear_var']

# منوی کشور سازنده
ttk.Label(root, text='کشور سازنده').grid(
    row=6, column=0, padx=20, pady=8, sticky='w'
)
origin_var = tk.StringVar(value='ژاپن')
ttk.Combobox(
    root,
    textvariable=origin_var,
    values=['آمریکا', 'اروپا', 'ژاپن'],
    state='readonly',
    width=12
).grid(row=6, column=1, padx=20, pady=8)

# دکمه پیش‌بینی
ttk.Button(
    root,
    text='پیش‌بینی',
    command=predict
).grid(row=7, column=0, columnspan=2, pady=15)

# نمایش نتیجه
result_label = ttk.Label(root, text='', font=('Arial', 12, 'bold'))
result_label.grid(row=8, column=0, columnspan=2, pady=10)

root.mainloop()