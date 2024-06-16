import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_data(file):
    data = pd.read_csv(file, index_col='Date', parse_dates=True)
    data = data.dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    joblib.dump(scaler, 'scaler.pkl')  # Save the fitted scaler
    return scaled_data, data['Close']

def retrain_model(X_train, y_train, best_params):
    model = LinearRegression(**best_params)
    model.fit(X_train, y_train)
    joblib.dump(model, 'optimized_model.pkl')
    return model

if __name__ == "__main__":
    X, y = prepare_data('AAPL.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_params = {'copy_X': True, 'fit_intercept': False, 'n_jobs': None, 'positive': False}
    model = retrain_model(X_train, y_train, best_params)
    print("Optimized model and scaler trained and saved successfully.")
