from prophet import Prophet
import pandas as pd

class ProphetService:
    def __init__(self):
        self.models = {}
        self.history = {}

    def train(self, symbol, history):
        df = pd.DataFrame(history)
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = pd.to_numeric(df["y"])
        df = df.sort_values("ds")

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.03
        )

        model.fit(df)

        self.models[symbol] = model
        self.history[symbol] = df

    def predict(self, symbol, periods):
        if symbol not in self.models:
            raise ValueError("Model not trained for this symbol")

        model = self.models[symbol]
        history_df = self.history[symbol]

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # 🔴 FORCE ISO STRING FORMAT
        history_out = history_df.copy()
        history_out["ds"] = history_out["ds"].dt.strftime("%Y-%m-%d")

        forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
        forecast_out["ds"] = forecast_out["ds"].dt.strftime("%Y-%m-%d")

        return {
            "history": history_out.to_dict(orient="records"),
            "forecast": forecast_out.to_dict(orient="records"),
        }

