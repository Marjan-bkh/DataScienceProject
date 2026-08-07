import joblib

model = joblib.load("ModelsOutcome/dow_jones_rf_model.pkl")
print(type(model))
print(model)