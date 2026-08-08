DATASETS = {
    "regression": {
        "bike_rental": {
            "display_name": "Bike Rental Demand (Capital Bikeshare)",
            "description": "Predicts the number of daily bike rentals based on weather conditions, "
                            "season, and calendar features using historical Capital Bikeshare data.",
            "model_path": "../Regression/ModelsOutcome/bike_rental_model.pkl",
            "features_path": "../Regression/ModelsOutcome/bike_rental_features.pkl",
            "needs_scaling": False,
            "scaler_path": None,
            "feature_specs": {
                # این بخش رو باهم دقیق پر می‌کنیم بر اساس لیست فیچرهای واقعی مدل
                "temp": {"type": "numeric", "min": -10.0, "max": 40.0, "label": "Temperature (°C)"},
                "hum": {"type": "numeric", "min": 0.0, "max": 100.0, "label": "Humidity (%)"},
                "windspeed": {"type": "numeric", "min": 0.0, "max": 70.0, "label": "Wind Speed (km/h)"},
                "season": {"type": "categorical", "options": ["Spring", "Summer", "Fall", "Winter"], "label": "Season"},
                "weekday": {"type": "categorical", "options": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "label": "Day of Week"},
            },
            "metrics": {"r2": None, "rmse": None},  # با خودت پر می‌کنیم
        }
    },
    "classification": {},
    "clustering": {},
}