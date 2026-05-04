"""
config.py
---------
Centralized configuration for the Stock Predictor project.
Edit these values to change global behaviour without touching any logic files.
"""

# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------
DEFAULT_TICKER: str = "GOOG"
DEFAULT_START_DATE: str = "2015-01-01"
DEFAULT_END_DATE: str = "2024-01-01"

# Fraction of data used for training (rest goes to test)
TRAIN_SPLIT: float = 0.80

# Number of past days the model looks back to make one prediction
LOOKBACK_WINDOW: int = 100

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
LSTM_UNITS_1: int = 128
LSTM_UNITS_2: int = 64
DROPOUT_RATE: float = 0.2
DENSE_UNITS: int = 25
OUTPUT_UNITS: int = 1

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
EPOCHS: int = 50
BATCH_SIZE: int = 32
VALIDATION_SPLIT: float = 0.10
LEARNING_RATE: float = 0.001

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_SAVE_PATH: str = "models/stock_lstm_model.keras"
SCALER_SAVE_PATH: str = "models/scaler.pkl"

# ---------------------------------------------------------------------------
# Technical indicator parameters
# ---------------------------------------------------------------------------
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BB_PERIOD: int = 20
BB_STD: int = 2

SMA_WINDOWS: list = [20, 50, 100, 200]
EMA_WINDOWS: list = [12, 26]
