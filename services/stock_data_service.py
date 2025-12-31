import yfinance as yf
import pandas as pd

class StockDataService:
    @staticmethod
    def fetch_history(symbol, start_date, end_date):
        """
        Fetch real stock data from Yahoo Finance
        Returns Prophet-ready JSON: ds, y
        """

        ticker = yf.Ticker(symbol)

        df = ticker.history(
            start=start_date,
            end=end_date,
            auto_adjust=True
        )

        if df.empty:
            raise ValueError("No data found for this symbol")

        df = df.reset_index()

        # Prophet expects ds (date) and y (value)
        df = df.rename(columns={
            "Date": "ds",
            "Close": "y"
        })

        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)

        return df[["ds", "y"]].to_dict(orient="records")
