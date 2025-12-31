from flask import Flask, request, jsonify
from flask_cors import CORS
from services.prophet_service import ProphetService
import os

app = Flask(__name__)
CORS(app)

service = ProphetService()

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Stock Prophet API running"})

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()

    symbol = data.get("symbol")
    history = data.get("history")

    if not symbol or not history:
        return jsonify({"error": "symbol and history required"}), 400

    try:
        service.train(symbol.upper(), history)
        return jsonify({"message": f"Model trained for {symbol.upper()}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["GET"])
def predict():
    symbol = request.args.get("symbol")
    periods = int(request.args.get("periods", 30))

    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    try:
        result = service.forecast(symbol.upper(), periods)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
