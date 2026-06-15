import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('train_data_us.csv')
df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

df_train, df_valid = train_test_split(df, test_size=0.25, random_state=12345)