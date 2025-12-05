from flask import Flask, request, jsonify
from prophet import Prophet
import pandas as pd
import json

app = Flask(__name__)

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    df = pd.DataFrame(data['history'])  # Expects [{"ds": "2023-01-01", "y": 123}, ...]

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=data.get('periods', 30))
    forecast = model.predict(future)

    response = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(data.get('periods', 30)).to_dict(orient='records')
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
