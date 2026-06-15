import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('train_data_us.csv')

df_train, df_valid = train_test_split(df, random_state=54321,
                                      test_size=0.25)  # haz la división... # haz la división de datos para el entrenamiento y la validación

features_train = df_train.drop(['last_price'], axis=1)
target_train = df_train['last_price'] / 100000
features_valid = df_valid.drop(['last_price'], axis=1)
target_valid = df_valid['last_price'] / 100000

best_error = 10000  # configura el inicio de RECM
best_est = 0
best_depth = 0
for est in range(10, 51, 10):
    for depth in range(1, 11):
        model = RandomForestRegressor(random_state=54321, n_estimators=est, max_depth=depth)

        model.fit(features_train, target_train)

        predictions_valid = model.predict(features_valid)

        error = mean_squared_error(target_valid, predictions_valid) ** 0.5
        print("Validación RECM para los n_estimators de", est, ", depth=", depth, "is", error)
        if error < best_error:  # guardamos la configuración del modelo si se logra el error más bajo
            best_error = error
            best_est = est
            best_depth = depth

print("RECM del mejor modelo en el conjunto de validación:", best_error, "n_estimators:", best_est, "best_depth:",
      best_depth)