"""CSV loader — reads stock_data/{area}/{ticker}.txt into pandas DataFrames."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import CSV_COLUMNS, DATA_DIR, DEFAULT_YEARS


class DataNotFoundError(Exception):
    """Raised when a ticker CSV is missing or empty."""


class NoDataInRangeError(Exception):
    """Raised when the requested date range has no bars."""


def _read_csv(filepath: str) -> pd.DataFrame:
    """Read a scraper-format CSV into a clean DataFrame.

    Expected format (no header, QUOTE_NONNUMERIC):
        "2021-07-19","2.91","2.99","2.65","2.98","650200"
    Columns: Date, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(
        filepath,
        header=None,
        names=CSV_COLUMNS,
        quoting=csv.QUOTE_NONNUMERIC,
        on_bad_lines="skip",
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def load_bars(
    ticker: str,
    area: str,
    start_range: Optional[str] = None,
    end_range: Optional[str] = None,
) -> pd.DataFrame:
    """Load historical bars for a ticker, optionally filtered by date range.

    Args:
        ticker: Stock symbol (e.g. "AAPL").
        area: Market area ("America", "SS", "SZ").
        start_range: Start date as "YYYY-MM-DD".  Defaults to 5 years ago.
        end_range: End date as "YYYY-MM-DD".  Defaults to today.

    Returns:
        DataFrame with Date index and columns: Close, High, Low, Open, Volume.

    Raises:
        DataNotFoundError: If the CSV file doesn't exist or is empty.
        NoDataInRangeError: If the date range contains no bars.
    """
    filepath = Path(DATA_DIR) / area / f"{ticker}.txt"

    if not filepath.exists():
        raise DataNotFoundError(
            f"Ticker not found: {ticker} in area {area} — "
            f"file does not exist at {filepath}"
        )

    df = _read_csv(str(filepath))

    if df.empty:
        raise DataNotFoundError(f"No data available for {ticker} in {area}")

    # Default range: 5 years of data
    today = datetime.now()
    if start_range is None:
        start_range = (today - timedelta(days=DEFAULT_YEARS * 365)).strftime("%Y-%m-%d")
    if end_range is None:
        end_range = today.strftime("%Y-%m-%d")

    start_ts = pd.Timestamp(start_range)
    end_ts = pd.Timestamp(end_range)

    filtered = df[(df.index >= start_ts) & (df.index <= end_ts)]

    if filtered.empty:
        raise NoDataInRangeError(
            f"No data for {ticker} between {start_range} and {end_range}"
        )

    return filtered


def resolve_date(df: pd.DataFrame, target_date: str) -> pd.Timestamp:
    """Find the next available trading day at or after *target_date*.

    Args:
        df: DataFrame with DatetimeIndex of trading days.
        target_date: Desired date as "YYYY-MM-DD".

    Returns:
        The Timestamp of the nearest trading day >= target_date.

    Raises:
        NoDataInRangeError: If no trading data exists on or after target_date.
    """
    target_ts = pd.Timestamp(target_date)
    available = df[df.index >= target_ts]
    if available.empty:
        raise NoDataInRangeError(
            f"No trading data on or after {target_date} for this ticker"
        )
    return available.index[0]
