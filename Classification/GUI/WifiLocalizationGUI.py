import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import joblib

# -------------------------------------------------------------------
# بارگذاری مدل و scaler ذخیره‌شده (همون فایل‌هایی که train_model.py ساخت)
# -------------------------------------------------------------------
MODEL_PATH = '../ModelsOutcome/wifi_room_model.pkl'
SCALER_PATH = '../ModelsOutcome/wifi_room_scaler.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

WIFI_COLS = ['wifi_1', 'wifi_2', 'wifi_3', 'wifi_4', 'wifi_5', 'wifi_6', 'wifi_7']

# باید دقیقاً همون ترتیب ستون‌هایی باشه که X_train_fe در train_model.py داشت
FEATURE_COLS = WIFI_COLS + ['strongest_signal', 'signal_range', 'strongest_router']


def build_feature_vector(raw_values):
    """
    raw_values: لیستی از ۷ عدد (قدرت سیگنال هر روتر) که کاربر وارد کرده
    خروجی: یک DataFrame تک‌ردیفی با همون اسم و ترتیب ستون‌هایی که
    scaler موقع fit_transform روی X_train_fe دیده بود.
    استفاده از DataFrame (به‌جای آرایه‌ی خام numpy) باعث میشه sklearn
    هشدار "X does not have valid feature names" رو نده.
    """
    arr = np.array(raw_values, dtype=float)

    sorted_vals = np.sort(arr)          # مرتب‌سازی از کوچک به بزرگ
    strongest_signal = sorted_vals[-1]  # بزرگترین مقدار = قوی‌ترین سیگنال
    signal_range = sorted_vals[-1] - sorted_vals[0]
    strongest_router = int(np.argmax(arr))  # اندیس روتر با قوی‌ترین سیگنال

    feature_row = list(arr) + [strongest_signal, signal_range, strongest_router]
    return pd.DataFrame([feature_row], columns=FEATURE_COLS)


def predict_room():
    try:
        raw_values = [float(entries[col].get()) for col in WIFI_COLS]
    except ValueError:
        messagebox.showerror("خطای ورودی", "لطفاً برای همه‌ی ۷ سیگنال یک عدد معتبر وارد کنید.")
        return

    feature_row = build_feature_vector(raw_values)
    feature_row_scaled = scaler.transform(feature_row)  # فقط transform، نه fit_transform - چون scaler از قبل روی train یاد گرفته شده

    predicted_room = model.predict(feature_row_scaled)[0]
    probabilities = model.predict_proba(feature_row_scaled)[0]
    confidence = probabilities[list(model.classes_).index(predicted_room)]

    result_var.set(f"اتاق پیش‌بینی‌شده: {predicted_room}   (اطمینان: {confidence*100:.1f}%)")


# -------------------------------------------------------------------
# ساخت پنجره‌ی اصلی
# -------------------------------------------------------------------
root = tk.Tk()
root.title("پیش‌بینی موقعیت مکانی بر اساس سیگنال وای‌فای")

# اندازه‌ی پویا بر اساس صفحه‌نمایش کاربر (به جای اندازه‌ی ثابت)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
win_width, win_height = 480, 420
pos_x = (screen_width - win_width) // 2
pos_y = (screen_height - win_height) // 2
root.geometry(f"{win_width}x{win_height}+{pos_x}+{pos_y}")
root.minsize(420, 400)

main_frame = ttk.LabelFrame(root, text="مقادیر قدرت سیگنال (dBm)", padding=15)
main_frame.pack(fill="both", expand=True, padx=15, pady=15)

entries = {}
for i, col in enumerate(WIFI_COLS):
    label = ttk.Label(main_frame, text=f"{col} :")
    label.grid(row=i, column=0, sticky="w", pady=5, padx=5)

    entry = ttk.Entry(main_frame, width=15)
    entry.grid(row=i, column=1, pady=5, padx=5)
    entry.insert(0, "-60")  # مقدار پیش‌فرض راهنما، صرفاً برای راحتی تست
    entries[col] = entry

predict_btn = ttk.Button(main_frame, text="پیش‌بینی اتاق", command=predict_room)
predict_btn.grid(row=len(WIFI_COLS), column=0, columnspan=2, pady=15)

result_var = tk.StringVar(value="")
result_label = ttk.Label(main_frame, textvariable=result_var, font=("Tahoma", 11, "bold"))
result_label.grid(row=len(WIFI_COLS) + 1, column=0, columnspan=2, pady=5)

root.mainloop()