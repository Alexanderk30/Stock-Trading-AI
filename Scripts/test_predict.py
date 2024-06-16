import requests
import json

url = 'http://127.0.0.1:5000/predict'
headers = {'Content-Type': 'application/json'}
data = [
    {
        "Open": 150.00,
        "High": 155.00,
        "Low": 149.00,
        "Close": 154.00,
        "Volume": 1000000
    },
    {
        "Open": 154.00,
        "High": 156.00,
        "Low": 153.00,
        "Close": 155.00,
        "Volume": 1100000
    }
]

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())
