import joblib
model = joblib.load('ModelsOutcome/online_news_popularity_gb_regressor.pkl')
print(model)

features = joblib.load('ModelsOutcome/online_news_popularity_reg_feature_columns.pkl')
print(features)
