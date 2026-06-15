import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error # ¡Asegúrate de que esto esté arriba!

df = pd.read_csv('train_data_us.csv')

features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 100000

# Inicializamos y entrenamos
final_model = RandomForestRegressor(random_state=54321, n_estimators=50, max_depth=10)
final_model.fit(features, target)

predictions = final_model.predict(features)

error = mean_squared_error(target, predictions) ** 0.5
print("mse:", error)