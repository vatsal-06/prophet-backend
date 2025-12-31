from flask import Flask, request, jsonify
from flask_cors import CORS
from services.prophet_service import ProphetService

app = Flask(__name__)
CORS(app)

prophet_service = ProphetService()

# -----------------------------
# Health Check
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Prophet Forecast API running"})


# -----------------------------
# Train Model
# -----------------------------
@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()

    if not data or "history" not in data:
        return jsonify({"error": "Missing 'history'"}), 400

    try:
        prophet_service.train(data["history"])
        return jsonify({
            "message": "Model trained successfully",
            "records_used": len(data["history"])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Forecast
# -----------------------------
@app.route("/forecast", methods=["GET"])
def forecast():
    try:
        periods = int(request.args.get("periods", 30))

        forecast_df = prophet_service.forecast(periods)

        response = {
            "forecast": forecast_df.tail(periods).to_dict(orient="records")
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Full Graph Data (Flutter)
# -----------------------------
@app.route("/graph-data", methods=["GET"])
def graph_data():
    try:
        periods = int(request.args.get("periods", 30))

        forecast_df = prophet_service.forecast(periods)

        response = {
            "history": prophet_service.history_df.to_dict(orient="records"),
            "forecast": forecast_df.tail(periods).to_dict(orient="records")
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
