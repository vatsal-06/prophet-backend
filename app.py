from flask import Flask, request, jsonify
from flask_cors import CORS
from services.prophet_service import ProphetService
from services.stock_data_service import StockDataService
import os

app = Flask(__name__)
CORS(app)

prophet_service = ProphetService()

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Stock Prophet API running"})

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()

    symbol = data.get("symbol")
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    if not symbol or not start_date or not end_date:
        return jsonify({"error": "symbol, start_date, end_date required"}), 400

    try:
        history = StockDataService.fetch_history(
            symbol=symbol.upper(),
            start_date=start_date,
            end_date=end_date
        )

        prophet_service.train(symbol.upper(), history)

        return jsonify({
            "message": f"Model trained for {symbol.upper()}",
            "records": len(history)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["GET"])
def predict():
    symbol = request.args.get("symbol")
    periods = int(request.args.get("periods", 30))

    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    try:
        result = prophet_service.predict(symbol.upper(), periods)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
