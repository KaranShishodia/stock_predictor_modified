# 📈 Stock Predictor Pro

A fully modular, production-quality stock price analysis and deep-learning forecasting system built with **Keras (LSTM)**, **yfinance**, and **Streamlit**.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Data** | Live OHLCV data via `yfinance` |
| **Indicators** | SMA, EMA, RSI, MACD, Bollinger Bands, Volume MA |
| **Model** | Stacked LSTM (128 → 64) + Dropout + Dense, saved as `.keras` |
| **Training** | Early stopping, LR reduction, model checkpointing |
| **Evaluation** | RMSE, MAE, MAPE, R² — printed and plotted |
| **Forecasting** | N-day rolling forward forecast |
| **Dashboard** | Streamlit app with interactive Plotly charts |
| **CLI** | Every module runnable as a standalone script |

---

## 🗂 Project Structure

```
stock_predictor/
├── config.py          # Central configuration (hyperparams, paths, etc.)
├── data.py            # Data fetching + feature engineering
├── model.py           # Keras LSTM architecture + save/load helpers
├── train.py           # Training pipeline
├── predict.py         # Back-test + future forecasting pipeline
├── evaluate.py        # Metrics (RMSE/MAE/MAPE/R²) + loss/prediction plots
├── visualize.py       # All matplotlib/seaborn charts
├── app.py             # Streamlit dashboard
├── requirements.txt
├── models/            # Saved .keras model + scaler.pkl (auto-created)
└── outputs/           # Generated PNG charts (auto-created)
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
# Default: GOOG, 2015-2024
python train.py

# Custom ticker and range
python train.py --ticker AAPL --start 2018-01-01 --end 2024-01-01 --epochs 100
```

Training will:
- Download data automatically
- Compute all technical indicators
- Train the LSTM with early stopping
- Save the model to `models/stock_lstm_model.keras`
- Save the scaler to `models/scaler.pkl`
- Print RMSE / MAE / MAPE / R² on the test set

### 3. Run predictions

```bash
# Back-test on held-out test data
python predict.py --ticker GOOG

# Back-test + 30-day future forecast
python predict.py --ticker GOOG --future_days 30
```

### 4. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ⚙️ Configuration

Edit `config.py` to change any global settings without touching logic:

```python
DEFAULT_TICKER    = "GOOG"
DEFAULT_START_DATE = "2015-01-01"
LOOKBACK_WINDOW   = 100   # past days fed into LSTM
LSTM_UNITS_1      = 128
LSTM_UNITS_2      = 64
EPOCHS            = 50
BATCH_SIZE        = 32
```

---

## 🧠 Model Architecture

```
Input  →  (LOOKBACK_WINDOW, 1)
           ↓
LSTM(128, return_sequences=True)  →  Dropout(0.2)
           ↓
LSTM(64,  return_sequences=False) →  Dropout(0.2)
           ↓
Dense(25, relu)
           ↓
Dense(1)       ← predicted next-day Close price (scaled)
```

**Loss:** Mean Squared Error (MSE)  
**Optimizer:** Adam (lr=0.001)  
**Callbacks:** EarlyStopping · ReduceLROnPlateau · ModelCheckpoint

---

## 📊 Technical Indicators

| Indicator | Parameters |
|---|---|
| SMA | 20, 50, 100, 200 days |
| EMA | 12, 26 days |
| RSI | 14-period |
| MACD | 12/26/9 |
| Bollinger Bands | 20-period, 2σ |
| Volume MA | 20-period |

---

## 📉 CLI Reference

```bash
# Data check
python data.py --ticker MSFT --start 2018-01-01 --end 2024-01-01

# Model summary
python model.py

# Train
python train.py --ticker TSLA --epochs 75

# Predict / forecast
python predict.py --ticker TSLA --future_days 20

# Streamlit app
streamlit run app.py
```

---

## ⚠️ Disclaimer

This project is **for educational purposes only**.  
Stock price predictions should not be used for real investment decisions.  
Past model performance does not guarantee future accuracy.

---

## 📦 Tech Stack

- [TensorFlow / Keras](https://keras.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/)

## Source Code
# model.py
import os
import config
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model as _keras_load_model

def build_lstm_model(
    lookback: int = config.LOOKBACK_WINDOW,
    lstm_units_1: int = config.LSTM_UNITS_1,
    lstm_units_2: int = config.LSTM_UNITS_2,
    dropout_rate: float = config.DROPOUT_RATE,
    dense_units: int = config.DENSE_UNITS,
    output_units: int = config.OUTPUT_UNITS,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    
    model = keras.Sequential(
        [
            # First LSTM block – returns full sequences for stacking
            layers.Input(shape=(lookback, 1), name="input_layer"),
            layers.LSTM(
                lstm_units_1,
                return_sequences=True,
                name="lstm_1",
            ),
            layers.Dropout(dropout_rate, name="dropout_1"),
            # Second LSTM block – returns only the last hidden state
            layers.LSTM(
                lstm_units_2,
                return_sequences=False,
                name="lstm_2",
            ),
            layers.Dropout(dropout_rate, name="dropout_2"),
            # Dense bottleneck
            layers.Dense(dense_units, activation="relu", name="dense_1"),
            # Output layer (linear activation for regression)
            layers.Dense(output_units, name="output"),
        ],
        name="StockLSTM",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    return model
def save_model(model: keras.Model, path: str = config.MODEL_SAVE_PATH) -> None:
   
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path)
    print(f"[model] Model saved to '{path}'.")


def load_model(path: str = config.MODEL_SAVE_PATH) -> keras.Model:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at '{path}'. "
            "Run train.py first to create one."
        )
    model = _keras_load_model(path)
    print(f"[model] Model loaded from '{path}'.")
    return model

if __name__ == "__main__":
    m = build_lstm_model()
    m.summary()

# train.py → training pipeline

"""
train.py
--------
Training pipeline for the Stock LSTM model.

Workflow
--------
1. Fetch & engineer features  (data.py)
2. Build the LSTM architecture (model.py)
3. Train with early-stopping & learning-rate reduction callbacks
4. Evaluate on the held-out test set
5. Save the trained model and scaler

Usage
-----
    python train.py
    python train.py --ticker MSFT --start 2015-01-01 --end 2024-01-01 --epochs 100
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow import keras

import config
from data import prepare_data
from evaluate import compute_metrics
from model import build_lstm_model, save_model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def get_callbacks(model_path: str = config.MODEL_SAVE_PATH) -> list:
    """Return a list of standard training callbacks.

    Callbacks
    ---------
    - **EarlyStopping** – halts training when ``val_loss`` stops improving
      for 10 consecutive epochs.  Restores the best weights automatically.
    - **ReduceLROnPlateau** – halves the learning rate when ``val_loss``
      plateaus for 5 epochs.
    - **ModelCheckpoint** – saves the best model to disk during training.
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    ticker: str = config.DEFAULT_TICKER,
    start_date: str = config.DEFAULT_START_DATE,
    end_date: str = config.DEFAULT_END_DATE,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lookback: int = config.LOOKBACK_WINDOW,
    model_path: str = config.MODEL_SAVE_PATH,
    scaler_path: str = config.SCALER_SAVE_PATH,
) -> dict:
    """Run the full training pipeline.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    start_date, end_date : str
        Date range for historical data.
    epochs : int
        Maximum number of training epochs (early-stopping may trigger sooner).
    batch_size : int
        Mini-batch size.
    lookback : int
        Sequence look-back window length.
    model_path : str
        Where to save the trained ``.keras`` model.
    scaler_path : str
        Where to save the fitted scaler.

    Returns
    -------
    dict
        ``history`` – Keras training history object
        ``metrics`` – evaluation metric dictionary
        ``data``    – the full prepared-data dictionary from data.py
    """
    # ------------------------------------------------------------------ data
    data = prepare_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        lookback=lookback,
        scaler_path=scaler_path,
    )

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    scaler = data["scaler"]

    print(
        f"\n[train] X_train: {X_train.shape}  y_train: {y_train.shape}"
        f"\n[train] X_test:  {X_test.shape}  y_test:  {y_test.shape}\n"
    )

    # --------------------------------------------------------------- build
    model = build_lstm_model(lookback=lookback)
    model.summary()

    # --------------------------------------------------------------- train
    print(f"\n[train] Starting training for up to {epochs} epochs...\n")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=get_callbacks(model_path),
        verbose=1,
    )

    # --------------------------------------------------------------- save
    save_model(model, model_path)

    # --------------------------------------------------------------- eval
    print("\n[train] Evaluating on test set...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_actual = data["y_test_raw"]

    metrics = compute_metrics(y_actual, y_pred)
    print(
        f"\n[train] ── Evaluation Results ──────────────────────────\n"
        f"         RMSE : {metrics['rmse']:.4f}\n"
        f"         MAE  : {metrics['mae']:.4f}\n"
        f"         MAPE : {metrics['mape']:.2f} %\n"
        f"         R²   : {metrics['r2']:.4f}\n"
        f"[train] ──────────────────────────────────────────────────\n"
    )

    return {"history": history, "metrics": metrics, "data": data}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Stock LSTM model")
    parser.add_argument(
        "--ticker", default=config.DEFAULT_TICKER, help="Stock ticker symbol"
    )
    parser.add_argument(
        "--start", default=config.DEFAULT_START_DATE, help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", default=config.DEFAULT_END_DATE, help="End date YYYY-MM-DD"
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS, help="Max training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size"
    )
    args = parser.parse_args()

    train(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

# predict.py → prediction logic
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

# evaluate.py → model evaluation
"""
evaluate.py
-----------
Evaluation metrics and helper functions.

Provides
--------
- ``compute_metrics``  – RMSE, MAE, MAPE, R²
- ``print_metrics``    – pretty-prints a metrics dict
- ``plot_training_history`` – loss / MAE curves saved to PNG

Usage (standalone)
------------------
    # Usually called from train.py or predict.py, not directly.
"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend for server / CI environments
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Actual values (original price scale).
    y_pred : array-like
        Predicted values (original price scale).

    Returns
    -------
    dict
        Keys: ``rmse``, ``mae``, ``mape``, ``r2``
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE – guard against division by zero
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def print_metrics(metrics: dict, ticker: str = "") -> None:
    """Pretty-print a metrics dictionary."""
    label = f" ({ticker})" if ticker else ""
    print(f"\n{'─'*50}")
    print(f"  Evaluation Metrics{label}")
    print(f"{'─'*50}")
    print(f"  RMSE  : {metrics['rmse']:.4f}")
    print(f"  MAE   : {metrics['mae']:.4f}")
    print(f"  MAPE  : {metrics['mape']:.2f} %")
    print(f"  R²    : {metrics['r2']:.4f}")
    print(f"{'─'*50}\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_history(
    history,
    save_path: Optional[str] = "outputs/training_history.png",
) -> None:
    """Plot and save training & validation loss / MAE curves.

    Parameters
    ----------
    history : keras.callbacks.History
        Object returned by ``model.fit()``.
    save_path : str or None
        Path to save the PNG.  Pass ``None`` to display interactively.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss", color="#0d6efd")
    if "val_loss" in history.history:
        axes[0].plot(
            history.history["val_loss"], label="Val Loss", color="#dc3545"
        )
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # MAE
    axes[1].plot(history.history["mae"], label="Train MAE", color="#198754")
    if "val_mae" in history.history:
        axes[1].plot(history.history["val_mae"], label="Val MAE", color="#fd7e14")
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Training history saved to '{save_path}'.")
    else:
        plt.show()

    plt.close(fig)


def plot_predictions(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    ticker: str = "",
    save_path: Optional[str] = "outputs/predictions_vs_actual.png",
) -> None:
    """Plot actual vs predicted prices and annotate with key metrics.

    Parameters
    ----------
    y_actual : np.ndarray
        True Close prices.
    y_pred : np.ndarray
        Model-predicted Close prices.
    ticker : str
        Stock ticker for the chart title.
    save_path : str or None
        PNG destination.  ``None`` shows interactively.
    """
    metrics = compute_metrics(y_actual, y_pred)
    annotation = (
        f"RMSE: {metrics['rmse']:.2f}   "
        f"MAE: {metrics['mae']:.2f}   "
        f"MAPE: {metrics['mape']:.2f}%   "
        f"R²: {metrics['r2']:.4f}"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_actual, label="Actual Price", color="#198754", linewidth=1.5)
    ax.plot(
        y_pred,
        label="Predicted Price",
        color="#dc3545",
        linewidth=1.5,
        linestyle="--",
    )
    ax.set_title(
        f"{ticker} — Actual vs Predicted Close Price",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Trading Day (test set)")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.annotate(
        annotation,
        xy=(0.5, 0.01),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        color="#555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Predictions plot saved to '{save_path}'.")
    else:
        plt.show()

    plt.close(fig)

# data.py → data loading and preprocessing

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

# visualize.py → plotting/visualization
"""
visualize.py
------------
All matplotlib / seaborn plotting helpers.

Functions
---------
- plot_price_history          – OHLCV history with volume bars
- plot_moving_averages        – Close price + SMA + EMA overlays
- plot_rsi                    – RSI sub-chart with overbought/oversold bands
- plot_macd                   – MACD / signal / histogram
- plot_bollinger_bands        – Price with Bollinger Band envelope
- plot_technical_indicators   – Comprehensive multi-panel summary chart
- plot_future_forecast        – Historical prices + N-day forward forecast
- plot_correlation_heatmap    – Feature correlation matrix

All functions accept an optional ``save_path`` argument.
Pass ``None`` to display interactively.
"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Common style defaults
plt.rcParams.update(
    {
        "figure.facecolor": "#0f0f23",
        "axes.facecolor": "#1a1a2e",
        "axes.edgecolor": "#444",
        "axes.labelcolor": "#ccc",
        "xtick.color": "#aaa",
        "ytick.color": "#aaa",
        "text.color": "#eee",
        "grid.color": "#333",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#555",
    }
)

# Color palette
BLUE = "#0d6efd"
GREEN = "#28a745"
RED = "#dc3545"
ORANGE = "#fd7e14"
PURPLE = "#6f42c1"
CYAN = "#17a2b8"


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------

def plot_price_history(
    df: pd.DataFrame,
    ticker: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot candlestick-style price history with volume bars.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Open, High, Low, Close, Volume columns.
    ticker : str
    save_path : str or None
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax1.plot(df.index, df["Close"], color=BLUE, linewidth=1.2, label="Close")
    ax1.fill_between(df.index, df["Low"], df["High"], alpha=0.12, color=BLUE)
    ax1.set_title(f"{ticker} — Price History", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True)

    colors = [GREEN if c >= o else RED for c, o in zip(df["Close"], df["Open"])]
    ax2.bar(df.index, df["Volume"], color=colors, width=1, alpha=0.7)
    ax2.set_ylabel("Volume")
    ax2.grid(True)

    plt.tight_layout()
    _save_or_show(fig, save_path, "[visualize] Price history saved")


def plot_moving_averages(
    df: pd.DataFrame,
    ticker: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot Close price with all SMA and EMA overlays."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df["Close"], color=BLUE, linewidth=1.2, label="Close", zorder=5)

    sma_colors = [RED, ORANGE, GREEN, PURPLE]
    for i, col in enumerate([c for c in df.columns if c.startswith("SMA_")]):
        ax.plot(df.index, df[col], linewidth=1, label=col, color=sma_colors[i % len(sma_colors)])

    ema_colors = [CYAN, "#e83e8c"]
    for i, col in enumerate([c for c in df.columns if c.startswith("EMA_")]):
        ax.plot(df.index, df[col], linewidth=1, linestyle="--", label=col, color=ema_colors[i % len(ema_colors)])

    ax.set_title(f"{ticker} — Price & Moving Averages", fontsize=13, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    _save_or_show(fig, save_path, "[visualize] Moving averages saved")


def plot_rsi(
    df: pd.DataFrame,
    ticker: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot the RSI indicator with overbought (70) and oversold (30) bands."""
    if "RSI" not in df.columns:
        print("[visualize] RSI column not found in DataFrame.")
        return

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(df.index, df["RSI"], color=ORANGE, linewidth=1.2, label="RSI")
    ax.axhline(70, color=RED, linestyle="--", linewidth=0.8, alpha=0.8, label="Overbought (70)")
    ax.axhline(30, color=GREEN, linestyle="--", linewidth=0.8, alpha=0.8, label="Oversold (30)")
    ax.axhline(50, color="#888", linestyle=":", linewidth=0.6, alpha=0.6)

    ax.fill_between(df.index, 70, df["RSI"], where=(df["RSI"] >= 70), alpha=0.2, color=RED)
    ax.fill_between(df.index, 30, df["RSI"], where=(df["RSI"] <= 30), alpha=0.2, color=GREEN)

    ax.set_title(f"{ticker} — RSI (14)", fontsize=13, fontweight="bold")
    ax.set_ylabel("RSI")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    _save_or_show(fig, save_path, "[visualize] RSI plot saved")


def plot_macd(
    df: pd.DataFrame,
    ticker: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot MACD line, signal line, and histogram."""
    if "MACD" not in df.columns:
        print("[visualize] MACD columns not found in DataFrame.")
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 1]}, sharex=True
    )

    ax1.plot(df.index, df["Close"], color=BLUE, linewidth=1, label="Close")
    ax1.set_title(f"{ticker} — MACD", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df.index, df["MACD"], color=BLUE, linewidth=1, label="MACD")
    ax2.plot(df.index, df["MACD_Signal"], color=RED, linewidth=1, label="Signal")

    hist = df["MACD_Hist"]
    colors = [GREEN if v >= 0 else RED for v in hist]
    ax2.bar(df.index, hist, color=colors, width=1, alpha=0.6, label="Histogram")
    ax2.axhline(0, color="#888", linewidth=0.8)
    ax2.set_ylabel("MACD")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    _save_or_show(fig, save_path, "[visualize] MACD plot saved")


def plot_bollinger_bands(
    df: pd.DataFrame,
    ticker: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot Bollinger Bands with the Close price inside the envelope."""
    if "BB_Upper" not in df.columns:
        print("[visualize] Bollinger Band columns not found in DataFrame.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df["Close"], color=BLUE, linewidth=1.2, label="Close", zorder=5)
    ax.plot(df.index, df["BB_Middle"], color=ORANGE, linewidth=1, linestyle="--", label="BB Middle")
    ax.plot(df.index, df["BB_Upper"], color=RED, linewidth=0.8, label="BB Upper")
    ax.plot(df.index, df["BB_Lower"], color=GREEN, linewidth=0.8, label="BB Lower")
    ax.fill_between(df.index, df["BB_Lower"], df["BB_Upper"], alpha=0.08, color=BLUE)

    ax.set_title(f"{ticker} — Bollinger Bands (20, 2σ)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    _save_or_show(fig, save_path, "[visualize] Bollinger Bands saved")


# ---------------------------------------------------------------------------
# Composite chart
# ---------------------------------------------------------------------------

def plot_technical_indicators(
    df: pd.DataFrame,
    ticker: str = "",
    save_path: Optional[str] = "outputs/technical_indicators.png",
) -> None:
    """Multi-panel summary: Price + MAs, Volume, RSI, MACD."""
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.08, height_ratios=[3, 1, 1.2, 1.2])

    ax_price = fig.add_subplot(gs[0])
    ax_vol   = fig.add_subplot(gs[1], sharex=ax_price)
    ax_rsi   = fig.add_subplot(gs[2], sharex=ax_price)
    ax_macd  = fig.add_subplot(gs[3], sharex=ax_price)

    # ── Price + MAs
    ax_price.plot(df.index, df["Close"], color=BLUE, linewidth=1.5, label="Close")
    for col, clr in zip(
        [c for c in df.columns if c.startswith("SMA_")],
        [RED, ORANGE, GREEN, PURPLE],
    ):
        ax_price.plot(df.index, df[col], linewidth=0.9, label=col, color=clr)
    ax_price.set_title(f"{ticker} — Technical Analysis Dashboard", fontsize=14, fontweight="bold", pad=10)
    ax_price.set_ylabel("Price (USD)")
    ax_price.legend(ncol=5, fontsize=8, loc="upper left")
    ax_price.grid(True)
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # ── Volume
    vol_colors = [GREEN if c >= o else RED for c, o in zip(df["Close"], df["Open"])]
    ax_vol.bar(df.index, df["Volume"], color=vol_colors, width=1, alpha=0.7)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(True)
    plt.setp(ax_vol.get_xticklabels(), visible=False)

    # ── RSI
    if "RSI" in df.columns:
        ax_rsi.plot(df.index, df["RSI"], color=ORANGE, linewidth=1)
        ax_rsi.axhline(70, color=RED, linestyle="--", linewidth=0.7, alpha=0.8)
        ax_rsi.axhline(30, color=GREEN, linestyle="--", linewidth=0.7, alpha=0.8)
        ax_rsi.fill_between(df.index, 70, df["RSI"], where=(df["RSI"] >= 70), alpha=0.15, color=RED)
        ax_rsi.fill_between(df.index, 30, df["RSI"], where=(df["RSI"] <= 30), alpha=0.15, color=GREEN)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.grid(True)
    plt.setp(ax_rsi.get_xticklabels(), visible=False)

    # ── MACD
    if "MACD" in df.columns:
        ax_macd.plot(df.index, df["MACD"], color=BLUE, linewidth=1, label="MACD")
        ax_macd.plot(df.index, df["MACD_Signal"], color=RED, linewidth=1, label="Signal")
        hist = df["MACD_Hist"]
        bar_colors = [GREEN if v >= 0 else RED for v in hist]
        ax_macd.bar(df.index, hist, color=bar_colors, width=1, alpha=0.55)
        ax_macd.axhline(0, color="#888", linewidth=0.7)
        ax_macd.set_ylabel("MACD")
        ax_macd.legend(fontsize=8)
        ax_macd.grid(True)

    _save_or_show(fig, save_path, "[visualize] Technical indicators chart saved")


# ---------------------------------------------------------------------------
# Future forecast chart
# ---------------------------------------------------------------------------

def plot_future_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    ticker: str = "",
    lookback_days: int = 200,
    save_path: Optional[str] = None,
) -> None:
    """Plot the last ``lookback_days`` of history + forward forecast.

    Parameters
    ----------
    historical_df : pd.DataFrame
        Full historical DataFrame (must have a ``Close`` column).
    forecast_df : pd.DataFrame
        DataFrame with ``Predicted_Close`` column, indexed by future dates.
    ticker : str
    lookback_days : int
        How many historical days to show before the forecast starts.
    save_path : str or None
    """
    hist_tail = historical_df["Close"].tail(lookback_days)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(hist_tail.index, hist_tail.values, color=BLUE, linewidth=1.5, label="Historical Close")

    # Connect history → forecast with a dotted bridge
    last_hist_date = hist_tail.index[-1]
    last_hist_price = hist_tail.values[-1]
    first_pred_date = forecast_df.index[0]
    first_pred_price = forecast_df["Predicted_Close"].iloc[0]
    ax.plot(
        [last_hist_date, first_pred_date],
        [last_hist_price, first_pred_price],
        color=ORANGE,
        linewidth=1.2,
        linestyle=":",
    )

    ax.plot(
        forecast_df.index,
        forecast_df["Predicted_Close"],
        color=ORANGE,
        linewidth=2,
        linestyle="--",
        label=f"Forecast ({len(forecast_df)}d)",
        marker="o",
        markersize=3,
    )

    ax.fill_between(forecast_df.index, forecast_df["Predicted_Close"], alpha=0.12, color=ORANGE)
    ax.axvline(last_hist_date, color="#888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(
        f"{ticker} — {len(forecast_df)}-Day Price Forecast",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    _save_or_show(fig, save_path, "[visualize] Future forecast saved")


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    ticker: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Seaborn correlation heatmap of all numeric features."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        linewidths=0.3,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"{ticker} — Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path, "[visualize] Correlation heatmap saved")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(fig, save_path: Optional[str], msg: str) -> None:
    """Save the figure to ``save_path`` or show it interactively."""
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"{msg} → '{save_path}'.")
    else:
        plt.show()
    plt.close(fig)

# app.py → likely the main app (UI or entry point)

"""
app.py
------
Streamlit dashboard for interactive stock analysis and prediction.

Run
---
    streamlit run app.py

Features
--------
- Sidebar controls: ticker, date range, forecast horizon
- Live data download via yfinance
- KPI cards: latest price, daily change, moving averages
- Interactive Plotly charts: price history, MAs, RSI, MACD, Bollinger Bands
- Model loading with graceful fallback
- Back-test predictions vs actual prices
- N-day future price forecast with confidence context
- Model metrics (RMSE, MAE, MAPE, R²) displayed in the UI
- Export predictions to CSV
"""

import os
import io
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from data import fetch_stock_data, add_all_indicators, split_data
from evaluate import compute_metrics
from model import load_model, build_lstm_model
from train import train as train_model

# ────────────────────────────────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal dark-theme CSS tweak
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 18px 22px;
        border-left: 4px solid;
        margin-bottom: 8px;
    }
    .metric-title { font-size: 13px; color: #aaa; margin-bottom: 4px; }
    .metric-value { font-size: 24px; font-weight: 700; color: #fff; }
    .metric-delta { font-size: 13px; margin-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")

    POPULAR_TICKERS = ["AAPL", "GOOG", "AMZN", "MSFT", "TSLA", "NVDA", "META", "NFLX", "Custom..."]
    selected = st.selectbox("Stock Ticker", options=POPULAR_TICKERS, index=1)
    if selected == "Custom...":
        ticker = st.text_input("Enter Ticker Symbol", value="", placeholder="e.g. UBER").upper().strip()
        if not ticker:
            st.warning("Please enter a ticker symbol.")
            st.stop()
    else:
        ticker = selected
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start", datetime.date(2015, 1, 1))
    with col_e:
        end_date = st.date_input("End", datetime.date(2024, 1, 1))

    st.markdown("---")
    st.subheader("Model")
    model_path = st.text_input(
        "Model Path (.keras)",
        value=config.MODEL_SAVE_PATH,
        help="Path to a trained .keras model file.",
    )
    future_days = st.slider("Forecast Horizon (days)", min_value=5, max_value=90, value=30)

    st.markdown("---")
    predict_btn = st.button("🚀 Run Prediction", use_container_width=True)
    forecast_btn = st.button("🔮 Forecast Future Prices", use_container_width=True)

    st.markdown("---")
    st.caption("📌 Powered by Keras · yfinance · Streamlit")


# ────────────────────────────────────────────────────────────────────────────
# Title
# ────────────────────────────────────────────────────────────────────────────
st.title("📈 Stock Predictor Pro")
st.markdown("*AI-powered stock price analysis and forecasting dashboard*")
st.markdown("---")


# ────────────────────────────────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
    df = fetch_stock_data(ticker, str(start), str(end))
    return add_all_indicators(df)


with st.spinner(f"Downloading data for **{ticker}**..."):
    try:
        df = get_data(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()

if df.empty:
    st.error("No data returned. Check the ticker symbol and date range.")
    st.stop()

st.success(f"✅ Loaded **{len(df):,}** trading days for **{ticker}**")


# ────────────────────────────────────────────────────────────────────────────
# KPI Cards
# ────────────────────────────────────────────────────────────────────────────
latest   = float(df["Close"].iloc[-1])
prev     = float(df["Close"].iloc[-2])
chg      = latest - prev
chg_pct  = (chg / prev) * 100
ma50     = float(df["SMA_50"].iloc[-1]) if "SMA_50" in df else float(df["Close"].rolling(50).mean().iloc[-1])
ma200    = float(df["SMA_200"].iloc[-1]) if "SMA_200" in df else float(df["Close"].rolling(200).mean().iloc[-1])
rsi_now  = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else None

arrow    = "▲" if chg >= 0 else "▼"
chg_clr  = "#28a745" if chg >= 0 else "#dc3545"

c1, c2, c3, c4, c5 = st.columns(5)

def kpi(col, title, value, delta="", border_color="#0d6efd", delta_color="#aaa"):
    col.markdown(
        f"""
        <div class="metric-card" style="border-color:{border_color}">
          <div class="metric-title">{title}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-delta" style="color:{delta_color}">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

kpi(c1, "Latest Close",  f"${latest:.2f}",  f"{arrow} {abs(chg):.2f} ({chg_pct:.2f}%)", "#0d6efd", chg_clr)
kpi(c2, "50-Day SMA",    f"${ma50:.2f}",     "Moving Average",    "#ffc107", "#aaa")
kpi(c3, "200-Day SMA",   f"${ma200:.2f}",    "Moving Average",    "#dc3545", "#aaa")
kpi(c4, "Data Points",   f"{len(df):,}",     f"{start_date} → {end_date}", "#6f42c1", "#aaa")
if rsi_now is not None:
    rsi_signal = "Overbought ⚠️" if rsi_now > 70 else ("Oversold 🟢" if rsi_now < 30 else "Neutral")
    kpi(c5, "RSI (14)", f"{rsi_now:.1f}", rsi_signal, "#17a2b8", "#aaa")

st.markdown("---")


# ────────────────────────────────────────────────────────────────────────────
# Charts – tabs
# ────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Price & MAs", "📉 RSI", "📈 MACD", "🎯 Bollinger Bands"]
)

# ── Tab 1: Price + Moving Averages
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#0d6efd", width=1.5)))
    for col, clr in zip(
        [c for c in df.columns if c.startswith("SMA_")],
        ["#dc3545", "#fd7e14", "#28a745", "#6f42c1"],
    ):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))
    fig.update_layout(
        title=f"{ticker} — Close Price & Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: RSI
with tab2:
    if "RSI" in df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#fd7e14")))
        fig_rsi.add_hline(y=70, line=dict(color="red",   dash="dash"), annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line=dict(color="green", dash="dash"), annotation_text="Oversold (30)")
        fig_rsi.update_layout(title=f"{ticker} — RSI (14)", template="plotly_dark", height=400, yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)
    else:
        st.info("RSI data not available.")

# ── Tab 3: MACD
with tab3:
    if "MACD" in df.columns:
        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.4, 0.6])
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#0d6efd")), row=1, col=1)
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#0d6efd")), row=2, col=1)
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#dc3545")), row=2, col=1)
        hist_colors = ["#28a745" if v >= 0 else "#dc3545" for v in df["MACD_Hist"]]
        fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color=hist_colors), row=2, col=1)
        fig_macd.update_layout(title=f"{ticker} — MACD", template="plotly_dark", height=550)
        st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.info("MACD data not available.")

# ── Tab 4: Bollinger Bands
with tab4:
    if "BB_Upper" in df.columns:
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(color="#dc3545", dash="dot")))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(color="#28a745", dash="dot"), fill="tonexty", fillcolor="rgba(13,110,253,0.06)"))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["BB_Middle"], name="BB Middle", line=dict(color="#fd7e14", dash="dash")))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#0d6efd", width=1.5)))
        fig_bb.update_layout(title=f"{ticker} — Bollinger Bands", template="plotly_dark", height=500)
        st.plotly_chart(fig_bb, use_container_width=True)
    else:
        st.info("Bollinger Band data not available.")


# ────────────────────────────────────────────────────────────────────────────
# Model — retrain whenever the ticker changes
# ────────────────────────────────────────────────────────────────────────────
if "model_cache" not in st.session_state:
    st.session_state.model_cache = {}   # { ticker: {"model": ..., "scaler": ...} }

model = None
scaler = None

cache_key = f"{ticker}_{str(start_date)}_{str(end_date)}"

if cache_key in st.session_state.model_cache:
    model  = st.session_state.model_cache[cache_key]["model"]
    scaler = st.session_state.model_cache[cache_key]["scaler"]
    st.sidebar.success(f"✅ Model ready ({ticker})")
else:
    st.sidebar.info(f"🔄 Training model for **{ticker}**…")
    train_placeholder = st.empty()
    with train_placeholder.container():
        st.info(f"⏳ Training LSTM for **{ticker}** — this takes ~1–2 minutes. Charts are live above.")
        prog = st.progress(0, text="Preparing data…")

    try:
        prog.progress(10, text="Fetching & engineering features…")
        ticker_model_path  = f"models/{ticker}_lstm.keras"
        ticker_scaler_path = f"models/{ticker}_scaler.pkl"

        prog.progress(25, text="Building & training LSTM…")
        result = train_model(
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date),
            model_path=ticker_model_path,
            scaler_path=ticker_scaler_path,
        )
        prog.progress(95, text="Finalising…")

        scaler = result["data"]["scaler"]
        model  = load_model(ticker_model_path)

        st.session_state.model_cache[cache_key] = {"model": model, "scaler": scaler}
        prog.progress(100, text="Done!")
        train_placeholder.empty()
        st.sidebar.success(f"✅ Model trained & ready ({ticker})")

    except Exception as e:
        train_placeholder.empty()
        st.sidebar.error(f"Training failed: {e}")
        st.error(f"❌ Could not train model for **{ticker}**: {e}")


# ────────────────────────────────────────────────────────────────────────────
# Back-test predictions
# ────────────────────────────────────────────────────────────────────────────
if predict_btn:
    if model is None or scaler is None:
        st.warning("No model loaded — cannot run predictions.")
    else:
        st.markdown("---")
        st.subheader("🤖 Back-test: Predicted vs Actual")

        with st.spinner("Running inference on test set..."):
            train_df, test_df = split_data(df)

            overlap = train_df["Close"].tail(config.LOOKBACK_WINDOW)
            test_close = pd.concat([overlap, test_df["Close"]])
            test_scaled = scaler.transform(test_close.values.reshape(-1, 1)).flatten()

            X_test, y_test_scaled = [], []
            for i in range(config.LOOKBACK_WINDOW, len(test_scaled)):
                X_test.append(test_scaled[i - config.LOOKBACK_WINDOW : i])
                y_test_scaled.append(test_scaled[i])

            X_test = np.array(X_test).reshape(-1, config.LOOKBACK_WINDOW, 1)
            y_pred_scaled = model.predict(X_test, verbose=0)
            y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
            y_actual = scaler.inverse_transform(np.array(y_test_scaled).reshape(-1, 1)).flatten()
            pred_dates = test_df.index[-len(y_actual):]

        metrics = compute_metrics(y_actual, y_pred)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE",  f"${metrics['rmse']:.2f}")
        m2.metric("MAE",   f"${metrics['mae']:.2f}")
        m3.metric("MAPE",  f"{metrics['mape']:.2f}%")
        m4.metric("R²",    f"{metrics['r2']:.4f}")

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=y_actual, name="Actual",    line=dict(color="#28a745", width=1.5)))
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=y_pred,   name="Predicted", line=dict(color="#dc3545", width=1.5, dash="dash")))
        fig_pred.update_layout(
            title=f"{ticker} — Actual vs Predicted Close Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=500,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Download button
        pred_df = pd.DataFrame({"Date": pred_dates, "Actual": y_actual, "Predicted": y_pred})
        csv_buf = io.StringIO()
        pred_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv_buf.getvalue(),
            file_name=f"{ticker}_predictions.csv",
            mime="text/csv",
        )


# ────────────────────────────────────────────────────────────────────────────
# Future forecast
# ────────────────────────────────────────────────────────────────────────────
if forecast_btn:
    if model is None or scaler is None:
        st.warning("No model loaded — cannot forecast.")
    else:
        st.markdown("---")
        st.subheader(f"🔮 {future_days}-Day Price Forecast")

        with st.spinner(f"Forecasting the next {future_days} trading days..."):
            train_df, test_df = split_data(df)

            seed = df["Close"].values[-config.LOOKBACK_WINDOW:]
            window = scaler.transform(seed.reshape(-1, 1)).flatten().tolist()

            preds_scaled = []
            for _ in range(future_days):
                x = np.array(window[-config.LOOKBACK_WINDOW:]).reshape(1, config.LOOKBACK_WINDOW, 1)
                p = model.predict(x, verbose=0)[0, 0]
                preds_scaled.append(p)
                window.append(p)

            preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
            last_date = df.index[-1]
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

        # Chart: last 200 days + forecast
        hist_tail = df["Close"].tail(200)
        fig_fut = go.Figure()
        fig_fut.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="Historical", line=dict(color="#0d6efd")))
        fig_fut.add_trace(go.Scatter(x=future_dates, y=preds, name="Forecast", line=dict(color="#fd7e14", dash="dash", width=2), fill="tozeroy", fillcolor="rgba(253,126,20,0.06)"))
        fig_fut.add_vline(x=str(last_date), line=dict(color="#888", dash="dot"))
        fig_fut.update_layout(
            title=f"{ticker} — {future_days}-Day Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=500,
        )
        st.plotly_chart(fig_fut, use_container_width=True)

        fcast_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": preds})
        st.dataframe(fcast_df.set_index("Date").style.format("${:.2f}"), use_container_width=True)

        csv_buf2 = io.StringIO()
        fcast_df.to_csv(csv_buf2, index=False)
        st.download_button(
            "⬇️ Download Forecast CSV",
            data=csv_buf2.getvalue(),
            file_name=f"{ticker}_forecast_{future_days}d.csv",
            mime="text/csv",
        )


# ────────────────────────────────────────────────────────────────────────────
# Raw data viewer
# ────────────────────────────────────────────────────────────────────────────
with st.expander("📋 View Raw Data & Indicators"):
    st.dataframe(df.tail(100).style.format("{:.4f}"), use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🔧 Stock Predictor Pro · Built with Streamlit, Keras & yfinance · For educational purposes only.")
