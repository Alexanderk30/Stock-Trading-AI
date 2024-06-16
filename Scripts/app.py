from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load('optimized_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Welcome to the Stock Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    return jsonify(prediction=prediction.tolist())

if __name__ == '__main__':
    app.run(port=5000, debug=True)
