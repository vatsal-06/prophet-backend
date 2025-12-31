import yfinance as yf
import pandas as pd

class StockDataService:
    @staticmethod
    def fetch_history(symbol, start_date, end_date):
        """
        Fetch real stock data from Yahoo Finance
        Returns Prophet-ready JSON (timezone-naive)
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

        # Rename columns for Prophet
        df = df.rename(columns={
            "Date": "ds",
            "Close": "y"
        })

        # 🔴 CRITICAL FIX: REMOVE TIMEZONE
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

        df["y"] = df["y"].astype(float)

        return df[["ds", "y"]].to_dict(orient="records")
