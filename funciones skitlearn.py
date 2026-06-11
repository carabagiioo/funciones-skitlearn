import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier


def error_count(answers, predictions):
    count = 0
    for i in range(len(answers)):
        if answers[i] != predictions[i]:
            count += 1
    return count


def accuracy(answers, predictions):
    # 1. Total de preguntas
    total = len(answers)

    # 2. Número de errores usando tu función anterior
    errores = error_count(answers, predictions)

    # 3. Calculamos la exactitud según la fórmula
    exactitud = (total - errores) / total

    return exactitud


'''''df = pd.read_csv('/datasets/train_data_us.csv')

df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345, class_weight='balanced')

model.fit(features, target)

test_df = pd.read_csv('/datasets/test_data_us.csv')

test_df.loc[test_df['last_price'] > 113000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 113000, 'price_class'] = 0

test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target = test_df['price_class']
test_predictions = model.predict(test_features)'''