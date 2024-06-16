import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def prepare_data(file):
    data = pd.read_csv(file, index_col='Date', parse_dates=True)
    data = data.dropna()
    scaler = joblib.load('scaler.pkl')  # Load the fitted scaler
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    return scaled_data, data['Close']

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R-squared: {r2}")

if __name__ == "__main__":
    X, y = prepare_data('AAPL.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load('optimized_model.pkl')
    evaluate_model(model, X_test, y_test)
