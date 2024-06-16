import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def prepare_data(file):
    data = pd.read_csv(file, index_col='Date', parse_dates=True)
    data = data.dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def optimize_hyperparameters(X, y):
    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'n_jobs': [None, -1],
        'positive': [True, False]
    }
    grid = GridSearchCV(LinearRegression(), param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_params_

if __name__ == "__main__":
    scaled_data = prepare_data('AAPL.csv')
    X = scaled_data[:, :-1]
    y = scaled_data[:, -1]
    best_params = optimize_hyperparameters(X, y)
    print(best_params)
