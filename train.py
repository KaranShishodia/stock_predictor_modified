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
