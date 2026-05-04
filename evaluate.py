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
