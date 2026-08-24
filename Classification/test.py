import os
import joblib

# مسیر پوشه ModelsOutcome برای یکی از دیتاست‌ها رو اینجا بذار
# مثلاً برای یک دیتاست classification:
folder_path = r"ModelsOutcome"  # مسیر واقعی خودت رو جایگزین کن

# لیست همه فایل‌های pkl توی این پوشه
files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
print("فایل‌های pkl موجود:")
for f in files:
    print(" -", f)

print("\n" + "=" * 50)

# حالا هر کدوم رو لود کن و نوعش رو ببین
for f in files:
    path = os.path.join(folder_path, f)
    obj = joblib.load(path)
    print(f"\nفایل: {f}")
    print(f"نوع: {type(obj)}")

    # اگه دیکشنری بود، کلیدهاش رو نشون بده
    if isinstance(obj, dict):
        print(f"کلیدها: {list(obj.keys())}")
    # اگه لیست بود (احتمالاً اسم فیچرها)
    elif isinstance(obj, list):
        print(f"محتوا (نمونه): {obj[:5]}")
    else:
        print(f"این احتمالاً خود مدل sklearn هست: {obj}")