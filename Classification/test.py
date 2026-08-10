import joblib
model = joblib.load("ModelsOutcome/bankruptcy_model.pkl")
print(model)
features = joblib.load("ModelsOutcome/bankruptcy_feature.pkl")
print(features)

encoder = joblib.load("ModelsOutcome/bankruptcy_risk_mapping.pkl")
print(encoder)