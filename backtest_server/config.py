"""Configuration for the backtest server.

Override via environment variables or by editing defaults here.
"""

import os
from pathlib import Path

# Paths
_BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.environ.get("BACKTEST_DATA_DIR", str(_BASE_DIR / "stock_data"))

# Date defaults
DEFAULT_YEARS = int(os.environ.get("BACKTEST_DEFAULT_YEARS", "5"))

# Server
HOST = os.environ.get("BACKTEST_HOST", "0.0.0.0")
PORT = int(os.environ.get("BACKTEST_PORT", "8000"))

# Backtest defaults
DEFAULT_CAPITAL = float(os.environ.get("BACKTEST_DEFAULT_CAPITAL", "100000.00"))
DEFAULT_COMMISSION = float(os.environ.get("BACKTEST_DEFAULT_COMMISSION", "0.005"))

# CSV column names (matching scrapper.py output — yfinance history() natural
# order: Date, Open, High, Low, Close, Volume)
CSV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
