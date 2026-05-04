"""
data.py
-------
Handles all data fetching, cleaning, feature engineering, and scaling.

Responsibilities
----------------
- Download OHLCV data via yfinance.
- Compute a rich set of technical indicators.
- Scale features for neural-network consumption.
- Build the supervised (X, y) sequences used for LSTM training / inference.

Usage (standalone)
------------------
    python data.py --ticker AAPL --start 2015-01-01 --end 2024-01-01
"""

import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

import config

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. ``"AAPL"``.
    start_date : str
        Inclusive start date, ``"YYYY-MM-DD"`` format.
    end_date : str
        Exclusive end date, ``"YYYY-MM-DD"`` format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume.

    Raises
    ------
    ValueError
        If yfinance returns an empty DataFrame (bad ticker or range).
    """
    print(f"[data] Downloading {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Check the symbol and date range."
        )

    # yfinance sometimes returns MultiIndex columns — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    print(f"[data] Downloaded {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def add_sma(df: pd.DataFrame) -> pd.DataFrame:
    """Add Simple Moving Average columns for each window in config."""
    for w in config.SMA_WINDOWS:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Add Exponential Moving Average columns for each window in config."""
    for w in config.EMA_WINDOWS:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = config.RSI_PERIOD) -> pd.DataFrame:
    """Compute the Relative Strength Index (RSI).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Close`` column.
    period : int
        Look-back period (default 14).
    """
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> pd.DataFrame:
    """Add MACD, MACD Signal, and MACD Histogram columns."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = config.BB_PERIOD,
    num_std: int = config.BB_STD,
) -> pd.DataFrame:
    """Add Bollinger Bands (upper, middle, lower) columns."""
    mid = df["Close"].rolling(window=period).mean()
    std = df["Close"].rolling(window=period).std()

    df["BB_Middle"] = mid
    df["BB_Upper"] = mid + num_std * std
    df["BB_Lower"] = mid - num_std * std
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based derived features."""
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function: apply every technical indicator in sequence.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with all indicator columns appended.
        Rows that still contain NaN (warm-up period) are dropped.
    """
    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_indicators(df)
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into training and testing sets (no shuffle).

    Parameters
    ----------
    df : pd.DataFrame
    train_ratio : float
        Proportion of rows allocated to training.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)``
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(
        f"[data] Train: {len(train_df)} rows | Test: {len(test_df)} rows"
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scaler(
    train_close: pd.Series,
    save_path: str = config.SCALER_SAVE_PATH,
) -> MinMaxScaler:
    """Fit a MinMaxScaler on the training Close prices and persist it.

    Parameters
    ----------
    train_close : pd.Series
        Training set Close prices.
    save_path : str
        File path to save the fitted scaler (pickle).

    Returns
    -------
    MinMaxScaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_close.values.reshape(-1, 1))

    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[data] Scaler saved to '{save_path}'.")
    return scaler


def load_scaler(path: str = config.SCALER_SAVE_PATH) -> MinMaxScaler:
    """Load a previously saved MinMaxScaler.

    Parameters
    ----------
    path : str
        Path to the pickled scaler.

    Returns
    -------
    MinMaxScaler
    """
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[data] Scaler loaded from '{path}'.")
    return scaler


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def build_sequences(
    scaled_values: np.ndarray,
    lookback: int = config.LOOKBACK_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 1-D scaled time-series into (X, y) LSTM sequences.

    Each sample ``X[i]`` contains ``lookback`` consecutive price values,
    and ``y[i]`` is the next value to predict.

    Parameters
    ----------
    scaled_values : np.ndarray, shape (n,) or (n, 1)
        Scaled Close prices.
    lookback : int
        Number of past timesteps in each input window.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``X`` has shape ``(samples, lookback, 1)``
        ``y`` has shape ``(samples,)``
    """
    values = scaled_values.flatten()
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback : i])
        y.append(values[i])

    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------

def prepare_data(
    ticker: str = config.DEFAULT_TICKER,
    start_date: str = config.DEFAULT_START_DATE,
    end_date: str = config.DEFAULT_END_DATE,
    lookback: int = config.LOOKBACK_WINDOW,
    scaler_path: str = config.SCALER_SAVE_PATH,
) -> dict:
    """End-to-end data pipeline: fetch → indicators → split → scale → sequences.

    Returns
    -------
    dict with keys:
        ``df``          – full enriched DataFrame
        ``train_df``    – training split
        ``test_df``     – test split
        ``scaler``      – fitted MinMaxScaler
        ``X_train``     – training sequences
        ``y_train``     – training labels
        ``X_test``      – test sequences
        ``y_test``      – test labels (scaled)
        ``y_test_raw``  – test labels (original price)
    """
    df = fetch_stock_data(ticker, start_date, end_date)
    df = add_all_indicators(df)

    train_df, test_df = split_data(df)

    scaler = fit_scaler(train_df["Close"], save_path=scaler_path)

    # Scale training Close prices
    train_scaled = scaler.transform(
        train_df["Close"].values.reshape(-1, 1)
    ).flatten()

    # For the test set we prepend the last `lookback` days from training
    # so that the first test prediction still has a full window.
    overlap = train_df["Close"].tail(lookback)
    test_close = pd.concat([overlap, test_df["Close"]])
    test_scaled = scaler.transform(
        test_close.values.reshape(-1, 1)
    ).flatten()

    X_train, y_train = build_sequences(train_scaled, lookback)
    X_test, y_test = build_sequences(test_scaled, lookback)

    y_test_raw = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_raw": y_test_raw,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch & prepare stock data")
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--start", default=config.DEFAULT_START_DATE)
    parser.add_argument("--end", default=config.DEFAULT_END_DATE)
    args = parser.parse_args()

    result = prepare_data(args.ticker, args.start, args.end)
    print("\n[data] Indicator columns:")
    print([c for c in result["df"].columns if c not in ("Open", "High", "Low", "Close", "Volume")])
    print(f"[data] X_train shape: {result['X_train'].shape}")
    print(f"[data] X_test shape:  {result['X_test'].shape}")
