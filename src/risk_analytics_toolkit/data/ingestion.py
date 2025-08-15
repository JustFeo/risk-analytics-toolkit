from __future__ import annotations



import pandas as pd
import yfinance as yf


def fetch_yfinance(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch OHLCV from yfinance and compute simple returns.

    Returns DataFrame with columns: Date (datetime), Close, Returns.
    """
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        raise ValueError("yfinance returned empty data")
    df = hist[["Close"]].copy()
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "Date"}, inplace=True)
    df["Returns"] = df["Close"].pct_change().fillna(0.0)
    return df[["Date", "Close", "Returns"]]


def load_csv(path: str, date_col: str, price_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(by=date_col, inplace=True)
    df["Returns"] = df[price_col].pct_change().fillna(0.0)
    return df.rename(columns={date_col: "Date", price_col: "Close"})[["Date", "Close", "Returns"]]


def load_claims_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"timestamp", "amount"}.issubset(df.columns):
        raise ValueError("claims csv must have 'timestamp' and 'amount'")
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # naive ok
    return df[["timestamp", "amount"]]
