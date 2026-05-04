"""
predict.py
----------
Prediction pipeline: load a trained model and generate future price forecasts.

Two prediction modes
--------------------
1. **Test-set evaluation** – compare predictions against held-out actual prices.
2. **Future forecasting** – iteratively predict N days beyond the latest data point.

Usage
-----
    # Back-test on the test split
    python predict.py --ticker GOOG

    # Forecast the next 30 trading days
    python predict.py --ticker GOOG --future_days 30

    # Use a custom model / date range
    python predict.py --ticker AAPL --start 2018-01-01 --end 2024-01-01 --future_days 15
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import config
from data import fetch_stock_data, add_all_indicators, fit_scaler, load_scaler, split_data
from evaluate import compute_metrics, plot_predictions, print_metrics
from model import load_model
from visualize import plot_technical_indicators, plot_future_forecast


# ---------------------------------------------------------------------------
# Back-test prediction
# ---------------------------------------------------------------------------

def predict_test_set(
    ticker: str = config.DEFAULT_TICKER,
    start_date: str = config.DEFAULT_START_DATE,
    end_date: str = config.DEFAULT_END_DATE,
    model_path: str = config.MODEL_SAVE_PATH,
    scaler_path: str = config.SCALER_SAVE_PATH,
    lookback: int = config.LOOKBACK_WINDOW,
    save_plots: bool = True,
) -> dict:
    """Run model inference on the held-out test set and compute metrics.

    Parameters
    ----------
    ticker : str
    start_date, end_date : str
    model_path : str
        Path to the ``.keras`` model file.
    scaler_path : str
        Path to the pickled scaler.
    lookback : int
    save_plots : bool
        If ``True``, PNG plots are written to the ``outputs/`` folder.

    Returns
    -------
    dict
        Keys: ``y_actual``, ``y_pred``, ``metrics``, ``dates``
    """
    # ------------------------------------------------------------------ load
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    # ------------------------------------------------------------------ data
    df = fetch_stock_data(ticker, start_date, end_date)
    df = add_all_indicators(df)
    train_df, test_df = split_data(df)

    # Build test sequences (prepend last `lookback` rows from train for context)
    overlap = train_df["Close"].tail(lookback)
    test_close = pd.concat([overlap, test_df["Close"]])
    test_scaled = scaler.transform(
        test_close.values.reshape(-1, 1)
    ).flatten()

    X_test, y_test_scaled = [], []
    for i in range(lookback, len(test_scaled)):
        X_test.append(test_scaled[i - lookback : i])
        y_test_scaled.append(test_scaled[i])

    X_test = np.array(X_test).reshape(-1, lookback, 1)

    # ---------------------------------------------------------------- predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_actual = scaler.inverse_transform(
        np.array(y_test_scaled).reshape(-1, 1)
    ).flatten()

    metrics = compute_metrics(y_actual, y_pred)
    print_metrics(metrics, ticker=ticker)

    # ------------------------------------------------------------------ plots
    if save_plots:
        pred_dates = test_df.index[-len(y_actual):]
        plot_predictions(
            y_actual,
            y_pred,
            ticker=ticker,
            save_path=f"outputs/{ticker}_predictions_vs_actual.png",
        )
        plot_technical_indicators(
            df,
            ticker=ticker,
            save_path=f"outputs/{ticker}_technical_indicators.png",
        )

    return {
        "y_actual": y_actual,
        "y_pred": y_pred,
        "metrics": metrics,
        "dates": test_df.index[-len(y_actual):],
    }


# ---------------------------------------------------------------------------
# Future forecasting
# ---------------------------------------------------------------------------

def predict_future(
    ticker: str = config.DEFAULT_TICKER,
    start_date: str = config.DEFAULT_START_DATE,
    end_date: str = config.DEFAULT_END_DATE,
    future_days: int = 30,
    model_path: str = config.MODEL_SAVE_PATH,
    scaler_path: str = config.SCALER_SAVE_PATH,
    lookback: int = config.LOOKBACK_WINDOW,
    save_plots: bool = True,
) -> pd.DataFrame:
    """Forecast stock prices for ``future_days`` beyond ``end_date``.

    The function uses a rolling approach: after each new prediction, the
    predicted value is appended to the input window so subsequent steps
    look one day further ahead.

    Parameters
    ----------
    ticker : str
    start_date, end_date : str
    future_days : int
        Number of trading days to forecast.
    model_path, scaler_path : str
    lookback : int
    save_plots : bool

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Date`` and ``Predicted_Close``.
    """
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    df = fetch_stock_data(ticker, start_date, end_date)
    df = add_all_indicators(df)

    # Seed window = last `lookback` Close prices from the dataset
    seed_prices = df["Close"].values[-lookback:]
    seed_scaled = scaler.transform(
        seed_prices.reshape(-1, 1)
    ).flatten().tolist()

    predictions_scaled = []
    window = seed_scaled.copy()

    print(f"\n[predict] Forecasting {future_days} days ahead for {ticker}...")
    for _ in range(future_days):
        x = np.array(window[-lookback:]).reshape(1, lookback, 1)
        pred_scaled = model.predict(x, verbose=0)[0, 0]
        predictions_scaled.append(pred_scaled)
        window.append(pred_scaled)

    # Inverse-transform to original price scale
    predictions = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    ).flatten()

    # Build a business-day date range starting the day after end_date
    last_date = df.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_days,
    )

    results_df = pd.DataFrame(
        {"Date": future_dates, "Predicted_Close": predictions}
    )
    results_df.set_index("Date", inplace=True)

    print("\n[predict] Future price forecast:")
    print(results_df.to_string())

    if save_plots:
        plot_future_forecast(
            df,
            results_df,
            ticker=ticker,
            save_path=f"outputs/{ticker}_future_forecast_{future_days}d.png",
        )

    return results_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate stock price predictions"
    )
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--start", default=config.DEFAULT_START_DATE)
    parser.add_argument("--end", default=config.DEFAULT_END_DATE)
    parser.add_argument(
        "--future_days",
        type=int,
        default=0,
        help="Number of future days to forecast (0 = back-test only)",
    )
    parser.add_argument(
        "--model", default=config.MODEL_SAVE_PATH, help="Path to .keras model"
    )
    args = parser.parse_args()

    # Always run back-test
    predict_test_set(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        model_path=args.model,
    )

    # Optionally run future forecast
    if args.future_days > 0:
        predict_future(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            future_days=args.future_days,
            model_path=args.model,
        )
