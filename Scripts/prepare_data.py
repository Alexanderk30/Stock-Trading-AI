import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_data(file):
    data = pd.read_csv(file, index_col='Date', parse_dates=True)
    data = data.dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

scaled_data = prepare_data('AAPL.csv')
print(scaled_data)
