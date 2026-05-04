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
