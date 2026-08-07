import joblib

model = joblib.load("ModelsOutcome/liver_disorder_kmeans_k2_model.pkl")
print(type(model))
print(model)