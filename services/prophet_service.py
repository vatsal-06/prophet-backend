from prophet import Prophet
import pandas as pd

class ProphetService:
    def __init__(self):
        self.model = None
        self.history_df = None

    def train(self, history):
        df = pd.DataFrame(history)

        # Enforce schema
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])

        self.history_df = df.copy()

        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        self.model.fit(df)

    def forecast(self, periods=30):
        if self.model is None:
            raise ValueError("Model not trained yet")

        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
