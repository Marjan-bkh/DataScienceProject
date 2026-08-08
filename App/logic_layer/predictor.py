import pandas as pd


def predict(model, feature_columns, scaler, user_inputs: dict):
    """
    user_inputs: dict مثل {"temp": 25.0, "hum": 60.0, "season": "Summer", ...}
    feature_columns: لیست ستون‌های نهایی که مدل باهاشون train شده (بعد از encoding)
    """
    # یک ردیف DataFrame با ترتیب دقیق feature_columns می‌سازیم
    input_df = pd.DataFrame([user_inputs])

    # اگه فیچرهای categorical نیاز به one-hot دارن، اینجا باید encode بشن
    # این بخش رو بر اساس نحوه دقیق preprocessing هر دیتاست باهم تکمیل می‌کنیم
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    if scaler is not None:
        input_df = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)

    prediction = model.predict(input_df)[0]
    return prediction