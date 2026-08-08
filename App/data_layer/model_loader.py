import joblib
import os


def load_model(model_path):
    return joblib.load(model_path)


def load_feature_columns(features_path):
    return joblib.load(features_path)


def load_scaler(scaler_path):
    if scaler_path is None or not os.path.exists(scaler_path):
        return None
    return joblib.load(scaler_path)