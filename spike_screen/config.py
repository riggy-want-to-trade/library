"""Configuration for the spike & drop screening service.

Override via environment variables or by editing defaults here.
"""

import os
from pathlib import Path

# Paths
_BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.environ.get("SPIKE_DATA_DIR", str(_BASE_DIR / "stock_data"))
REPORT_DIR = Path(os.environ.get("SPIKE_REPORT_DIR", str(_BASE_DIR / "reports")))

# Server
HOST = os.environ.get("SPIKE_HOST", "0.0.0.0")
PORT = int(os.environ.get("SPIKE_PORT", "8001"))
MAX_WORKERS = int(os.environ.get("SPIKE_MAX_WORKERS", "8"))

# Screening defaults — thresholds are in PERCENT units (25.0 == 25%)
DEFAULT_WEEK_PCT = 25.0       # week-window spike threshold
DEFAULT_WEEK_DAYS = 5         # trading days in "a week"
DEFAULT_CUM_DAYS = 3          # consecutive days for the cum3 modes
DEFAULT_SMALL_PCT = 10.0      # threshold for the three cum3 modes
DEFAULT_SPAN_DAYS = 10        # 2-week span (trading days) for cum3_any3
DEFAULT_DROP_PCT = 15.0       # post-peak fall threshold
DEFAULT_POST_PEAK_DAYS = 10   # trading days after the peak for the drop
DEFAULT_EXCHANGES = ("SS", "SZ")
DEFAULT_MIN_BARS = 20         # minimum rows to screen a stock at all
DEFAULT_PER_STOCK = "max"     # which qualifying window to report per (stock, mode)

# CSV column order — matches scrapper.py output (yfinance history() natural
# order). NOTE: backtest_server/config.py has Open/Close swapped.
CSV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Spike screen modes
MODE_WEEK = "week_pct25"
MODE_CONSEC = "cum3_consecutive"
MODE_ANY3 = "cum3_any3"
MODE_STRICT = "cum3_strict"
MODES = (MODE_WEEK, MODE_CONSEC, MODE_ANY3, MODE_STRICT)

# B-share ticker prefixes (SS 900xxx = USD, SZ 200xxx/201xxx = HKD)
B_SHARE_PREFIXES = {"SS": ("900",), "SZ": ("200", "201")}

# Day-over-day Close ratio thresholds that start a new series segment.
# Real trading (10%/20% price limits, IPO/resumption no-limit days) never
# moves close-to-close by more than ~2x; bigger jumps are bad data rows or
# dividend/split back-adjustment discontinuities in the feed.
SEGMENT_COL = "_seg"
DAY_JUMP_HIGH = 2.0
DAY_JUMP_LOW = 0.5
