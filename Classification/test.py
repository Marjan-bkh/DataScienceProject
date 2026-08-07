import joblib

model = joblib.load("ModelsOutcome/adult_income_rf_model.pkl")
print(type(model))
print(model)