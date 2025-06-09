from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
import pandas as pd

app = Flask(__name__)
CORS(app)  # Optional but recommended if your frontend (e.g., Flutter) is separate

@app.route('/')
def home():
    return "✅ Prophet Forecast API is running!"

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'GET':
        return "Send a POST request with JSON time series data."

    try:
        data = request.get_json()
        history = data.get("history")
        periods = data.get("periods", 30)

        if not history:
            return jsonify({"error": "Missing 'history' data"}), 400

        df = pd.DataFrame(history)

        # Ensure required columns are present
        if 'ds' not in df.columns or 'y' not in df.columns:
            return jsonify({"error": "Data must contain 'ds' and 'y' columns"}), 400

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict(orient='records')
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
