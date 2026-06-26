import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv('/datasets/travel_insurance_us.csv')

encoder = OrdinalEncoder()
encoder.fit(data)
data_ordinal = encoder.transform(data)
data_ordinal = pd.DataFrame(encoder.transform(data), columns=data.columns)
'''data_ordinal = pd.DataFrame(encoder.fit_transform(data), 
                                                      columns=data.columns), tambien se puede usar esto, y combina el transform y fit'''
print(data_ordinal.head())