from prophet import Prophet
import pandas as pd

class ProphetService:
    def __init__(self):
        self.models = {}        # symbol -> model
        self.history = {}       # symbol -> dataframe

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

    def forecast(self, symbol, periods):
        if symbol not in self.models:
            raise ValueError("Model not trained for this symbol")

        model = self.models[symbol]
        history_df = self.history[symbol]

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        return {
            "history": history_df.to_dict(orient="records"),
            "forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                        .tail(periods)
                        .to_dict(orient="records")
        }
