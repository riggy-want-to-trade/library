"""CSV loader — reads stock_data/{area}/{ticker}.txt into pandas DataFrames.

File format (written by scrapper.py, no header, QUOTE_NONNUMERIC):
    "2021-09-24","7.36","7.36","7.28","7.33","39925596"
Columns: Date, Open, High, Low, Close, Volume

NOTE: this is Open-first (yfinance history() natural order) — backtest_server
reads these files with Open/Close swapped; this loader uses the corrected order.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .config import CSV_COLUMNS, DATA_DIR, DAY_JUMP_HIGH, DAY_JUMP_LOW, SEGMENT_COL


class DataNotFoundError(Exception):
    """Raised when a ticker CSV is missing or empty."""


class NoDataInRangeError(Exception):
    """Raised when the requested date range has no bars."""


@dataclass
class ScanBundle:
    """Result of a parallel scan over many tickers."""

    frames: list[tuple[str, str, pd.DataFrame]] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)  # {area, ticker, reason}


def _read_csv(filepath: str) -> pd.DataFrame:
    """Read a scraper-format CSV into a clean DataFrame.

    Expected format (no header, QUOTE_NONNUMERIC):
        "2021-09-24","7.36","7.36","7.28","7.33","39925596"
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
    # Hardening: duplicate dates keep the first occurrence
    df = df[~df.index.duplicated(keep="first")]
    # Drop rows with non-positive OHLC prices (would break ratio math)
    for col in ("Open", "High", "Low", "Close"):
        df = df[df[col] > 0]
    return df


def _add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with a segment id (SEGMENT_COL); the id increments at
    day-over-day Close discontinuities so price windows never mix regimes
    (bad feed rows, adjustment artifacts, suspension resumptions)."""
    close = df["Close"].to_numpy(dtype=float)
    ratio = np.empty(len(df))
    ratio[0] = 1.0
    ratio[1:] = close[1:] / close[:-1]
    breaks = (ratio > DAY_JUMP_HIGH) | (ratio < DAY_JUMP_LOW)
    df[SEGMENT_COL] = np.cumsum(breaks)
    return df


def load_stock_df(
    ticker: str,
    area: str,
    start_range: Optional[str] = None,
    end_range: Optional[str] = None,
) -> pd.DataFrame:
    """Load historical bars for a ticker, optionally filtered by date range.

    Args:
        ticker: Stock symbol with suffix (e.g. "600000.SS").
        area: Market area ("SS" or "SZ").
        start_range: Start date as "YYYY-MM-DD". Defaults to 5 years ago.
        end_range: End date as "YYYY-MM-DD". Defaults to today.

    Returns:
        DataFrame with Date index and columns: Open, High, Low, Close, Volume.

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

    today = datetime.now()
    if start_range is None:
        start_range = (today - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    if end_range is None:
        end_range = today.strftime("%Y-%m-%d")

    start_ts = pd.Timestamp(start_range)
    end_ts = pd.Timestamp(end_range)

    filtered = df[(df.index >= start_ts) & (df.index <= end_ts)]

    if filtered.empty:
        raise NoDataInRangeError(
            f"No data for {ticker} between {start_range} and {end_range}"
        )

    return _add_segments(filtered)


def _discover_stems(area: str) -> list[str]:
    """Sorted filename stems (e.g. "600000.SS") of DATA_DIR/{area}/*.txt."""
    data_path = Path(DATA_DIR) / area
    if not data_path.is_dir():
        raise ValueError(f"Unknown area '{area}' — directory missing: {data_path}")
    return sorted(f.stem for f in data_path.glob("*.txt") if f.is_file())


def discover_tickers(
    area: str, tickers: Optional[Sequence[str]] = None
) -> list[str]:
    """Discover tickers for an area, optionally filtered by a request list.

    Request entries may be bare codes ("600000") or with suffix ("600000.SS");
    matching is case-insensitive.

    Raises:
        ValueError: If the area directory doesn't exist.
    """
    stems = _discover_stems(area)
    if not tickers:
        return stems
    wanted = set()
    for t in tickers:
        t = t.strip().upper()
        if t.endswith(f".{area}"):
            t = t[: -len(area) - 1]
        wanted.add(t)
    return [s for s in stems if s.upper().split(".")[0] in wanted]


def count_tickers(area: str) -> int:
    """Number of ticker files in an area."""
    return len(_discover_stems(area))


def is_b_share(area: str, ticker: str) -> bool:
    """True for B-share tickers (SS 900xxx = USD, SZ 200xxx/201xxx = HKD)."""
    code = ticker.split(".")[0]
    from .config import B_SHARE_PREFIXES

    return any(code.startswith(p) for p in B_SHARE_PREFIXES.get(area, ()))


def load_all(
    areas: Sequence[str],
    start: Optional[str],
    end: Optional[str],
    tickers: Optional[Sequence[str]] = None,
    max_workers: int = 8,
    exclude_b_shares: bool = False,
) -> ScanBundle:
    """Load every ticker in the given areas (optionally filtered), in parallel.

    Skips (missing/empty/unreadable/no data in range) are collected in
    ``ScanBundle.skipped`` rather than raised.
    """
    bundle = ScanBundle()
    jobs = []
    for area in areas:
        for ticker in discover_tickers(area, tickers):
            if exclude_b_shares and is_b_share(area, ticker):
                continue
            jobs.append((area, ticker))

    if not jobs:
        return bundle

    def _one(job):
        area, ticker = job
        try:
            df = load_stock_df(ticker, area, start, end)
            return ("ok", area, ticker, df)
        except DataNotFoundError as e:
            return ("skip", area, ticker, "missing or empty file")
        except NoDataInRangeError:
            return ("skip", area, ticker, "no data in range")
        except Exception:
            return ("skip", area, ticker, "unreadable")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        for fut in as_completed(futures):
            status, area, ticker, payload = fut.result()
            if status == "ok":
                bundle.frames.append((area, ticker, payload))
            else:
                bundle.skipped.append({"area": area, "ticker": ticker, "reason": payload})

    bundle.frames.sort(key=lambda t: (t[0], t[1]))
    return bundle
