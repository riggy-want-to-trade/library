"""Screening logic — pure pandas/numpy functions, no I/O, no FastAPI.

All windows roll over ROWS (trading days), never calendar days, which
handles A-share suspension gaps correctly: a "week" is N trading rows.

Thresholds in ScreenParams are in PERCENT units (25.0 == 25%).

Performance note: window maxima/argmax are computed vectorized via
sliding_window_view (no per-window DataFrame slicing), and the per_stock
policy is applied INSIDE find_spikes so "max"/"first" build only the one
record that is actually returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .config import (
    DEFAULT_CUM_DAYS,
    DEFAULT_DROP_PCT,
    DEFAULT_MIN_BARS,
    DEFAULT_PER_STOCK,
    DEFAULT_POST_PEAK_DAYS,
    DEFAULT_SMALL_PCT,
    DEFAULT_SPAN_DAYS,
    DEFAULT_WEEK_DAYS,
    DEFAULT_WEEK_PCT,
    MODE_ANY3,
    MODE_CONSEC,
    MODE_STRICT,
    MODE_WEEK,
    MODES,
    SEGMENT_COL,
)


@dataclass(frozen=True)
class ScreenParams:
    """Screening thresholds. Percent fields are in percent units."""

    week_pct: float = DEFAULT_WEEK_PCT        # week-window spike threshold (%)
    week_days: int = DEFAULT_WEEK_DAYS        # trading days in "a week"
    cum_days: int = DEFAULT_CUM_DAYS          # consecutive days for cum3 modes
    small_pct: float = DEFAULT_SMALL_PCT      # cum3 modes threshold (%)
    span_days: int = DEFAULT_SPAN_DAYS        # 2-week span for cum3_any3 (trading days)
    drop_pct: float = DEFAULT_DROP_PCT        # post-peak fall threshold (%)
    post_peak_days: int = DEFAULT_POST_PEAK_DAYS  # trading days after peak
    per_stock: str = DEFAULT_PER_STOCK        # "max" | "first" | "all"
    min_bars: int = DEFAULT_MIN_BARS          # minimum rows to screen a stock


@dataclass(frozen=True)
class SpikeWindow:
    mode: str
    window_start: str      # YYYY-MM-DD, first trading day of the window/span
    window_end: str        # YYYY-MM-DD, last trading day of the window/span
    start_open: float      # Open of the first window day
    peak_date: str         # date of the max High within the window/span
    peak_high: float       # that max High
    gain_pct: float        # (peak_high / start_open - 1) * 100, rounded 2 dp
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DropResult:
    peak_date: str
    peak_high: float
    trough_date: str       # date of min Close in the post-peak window
    trough_close: float
    drop_pct: float        # (peak_high - trough_close) / peak_high * 100
    days_peak_to_trough: int  # calendar days
    trading_days: int      # trading rows from peak to trough


@dataclass(frozen=True)
class StockResult:
    area: str
    ticker: str
    spikes: dict = field(default_factory=dict)   # mode -> list[SpikeWindow]
    drop: Optional[DropResult] = None            # standalone global-peak drop
    pairs: dict = field(default_factory=dict)    # mode -> (SpikeWindow, DropResult)
    skipped_reason: Optional[str] = None


# ── Internal helpers ───────────────────────────────────────────────────

def _ratio(df: pd.DataFrame) -> np.ndarray:
    """Per-day (High - Open) / Open as a float array (fractions)."""
    high = df["High"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    return (high - open_) / open_


def _same_segment_mask(df: pd.DataFrame, width: int) -> Optional[np.ndarray]:
    """Boolean mask over window-START positions: True when the whole
    `width`-row window lies within one series segment (no discontinuity
    crossed). None when the frame has no segment column."""
    if SEGMENT_COL not in df.columns:
        return None
    seg = df[SEGMENT_COL].to_numpy()
    return seg[: len(seg) - width + 1] == seg[width - 1 :]


def _keep_ends(
    ends: np.ndarray, values: np.ndarray, starts_pos: np.ndarray, keep: str
) -> np.ndarray:
    """Reduce qualifying window-end positions per the keep policy.

    values is the per-window gain array indexed by window START position
    (same length as starts_pos).
    """
    if len(ends) == 0:
        return ends
    if keep == "first":
        return ends[:1]
    if keep == "max":
        best = int(values[starts_pos].argmax())
        return ends[best : best + 1]
    return ends  # "all"


def _fmt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def _day_dict(df: pd.DataFrame, i: int, r: np.ndarray) -> dict:
    return {
        "date": _fmt(df.index[i]),
        "open": float(df["Open"].iloc[i]),
        "high": float(df["High"].iloc[i]),
        "day_gain_pct": round(float(r[i]) * 100.0, 2),
    }


def _rolling_records(
    df: pd.DataFrame, pct: float, width: int, mode: str, keep: str
) -> list[SpikeWindow]:
    """Windows of `width` consecutive trading rows where
    (max High in window / Open of first window day - 1) >= pct (fraction).
    Serves week_pct25 and cum3_consecutive.
    """
    n = len(df)
    if n < width:
        return []
    high = df["High"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    sw = sliding_window_view(high, width)         # (n-width+1, width)
    hmax = sw.max(axis=1)
    peak_off = sw.argmax(axis=1)
    starts = open_[: n - width + 1]
    gain = hmax / starts - 1.0
    same = _same_segment_mask(df, width)
    if same is not None:
        gain[~same] = -np.inf  # windows crossing a discontinuity never qualify
    starts_pos = np.flatnonzero(np.isfinite(gain) & (gain >= pct))
    ends = starts_pos + (width - 1)
    ends = _keep_ends(ends, gain, starts_pos, keep)

    index = df.index
    records = []
    for i in ends:
        s = i - width + 1
        records.append(
            SpikeWindow(
                mode=mode,
                window_start=_fmt(index[s]),
                window_end=_fmt(index[i]),
                start_open=float(starts[s]),
                peak_date=_fmt(index[s + int(peak_off[s])]),
                peak_high=float(hmax[s]),
                gain_pct=round(float(gain[s]) * 100.0, 2),
            )
        )
    return records


def _strict_records(
    df: pd.DataFrame, r: np.ndarray, pct: float, cum_days: int, keep: str
) -> list[SpikeWindow]:
    """`cum_days` consecutive days where EACH day has (High-Open)/Open >= pct."""
    n = len(df)
    if n < cum_days:
        return []
    ok = np.isfinite(r) & (r >= pct)
    conv = np.convolve(ok.astype(float), np.ones(cum_days), mode="valid")
    same = _same_segment_mask(df, cum_days)
    if same is not None:
        conv[~same] = 0.0  # windows crossing a discontinuity never qualify
    ends = np.flatnonzero(conv == cum_days) + (cum_days - 1)
    if len(ends) == 0:
        return []

    high = df["High"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    sw = sliding_window_view(high, cum_days)
    hmax = sw.max(axis=1)
    peak_off = sw.argmax(axis=1)
    starts = open_[: n - cum_days + 1]
    gain = hmax / starts - 1.0
    starts_pos = ends - (cum_days - 1)
    ends = _keep_ends(ends, gain, starts_pos, keep)

    index = df.index
    records = []
    for i in ends:
        s = i - cum_days + 1
        days = [_day_dict(df, j, r) for j in range(s, i + 1)]
        records.append(
            SpikeWindow(
                mode=MODE_STRICT,
                window_start=_fmt(index[s]),
                window_end=_fmt(index[i]),
                start_open=float(starts[s]),
                peak_date=_fmt(index[s + int(peak_off[s])]),
                peak_high=float(hmax[s]),
                gain_pct=round(float(gain[s]) * 100.0, 2),
                detail={"days": days},
            )
        )
    return records


def _any3_records(
    df: pd.DataFrame, r: np.ndarray, pct: float, span_days: int, keep: str
) -> list[SpikeWindow]:
    """Any 3 days inside a `span_days`-row span whose daily
    (High - Open) / Open values sum to >= pct (fraction).

    Uses sliding_window_view + per-span top-3 sum (pandas 3.0.5 has no
    rolling().rank(method="first"), and rank-masking doesn't compute the
    top-3 sum anyway).
    """
    n = len(df)
    if n < span_days:
        return []
    sw = sliding_window_view(r, span_days)          # (n-span+1, span_days)
    top3 = np.sort(sw, axis=1)[:, -3:].sum(axis=1)  # sum of 3 largest day gains
    same = _same_segment_mask(df, span_days)
    if same is not None:
        top3[~same] = -np.inf  # spans crossing a discontinuity never qualify
    ends = np.flatnonzero(np.isfinite(top3) & (top3 >= pct)) + (span_days - 1)
    starts_pos = ends - (span_days - 1)
    ends = _keep_ends(ends, top3, starts_pos, keep)
    if len(ends) == 0:
        return []

    high = df["High"].to_numpy(dtype=float)
    sw_h = sliding_window_view(high, span_days)
    hmax = sw_h.max(axis=1)
    peak_off = sw_h.argmax(axis=1)
    open_ = df["Open"].to_numpy(dtype=float)
    index = df.index

    records = []
    for i in ends:
        s = i - span_days + 1
        top_pos = np.argsort(r[s : i + 1])[::-1][:3]
        days = [_day_dict(df, s + j, r) for j in sorted(top_pos)]
        records.append(
            SpikeWindow(
                mode=MODE_ANY3,
                window_start=_fmt(index[s]),
                window_end=_fmt(index[i]),
                start_open=float(open_[s]),
                peak_date=_fmt(index[s + int(peak_off[s])]),
                peak_high=float(hmax[s]),
                gain_pct=round(float(top3[s]) * 100.0, 2),
                detail={"days": days},
            )
        )
    return records


# ── Public screening functions ─────────────────────────────────────────

def find_spikes(df: pd.DataFrame, p: ScreenParams) -> dict:
    """Find qualifying spike windows for every mode.

    Returns {mode: [SpikeWindow, ...]} — empty lists when nothing
    qualifies (or the stock has fewer than p.min_bars rows). The
    p.per_stock policy is applied here ("max"/"first" -> at most one
    window per mode).
    """
    result = {m: [] for m in MODES}
    n = len(df)
    if n < p.min_bars:
        return result
    keep = p.per_stock
    r = _ratio(df)
    result[MODE_WEEK] = _rolling_records(df, p.week_pct / 100.0, p.week_days, MODE_WEEK, keep)
    result[MODE_CONSEC] = _rolling_records(df, p.small_pct / 100.0, p.cum_days, MODE_CONSEC, keep)
    result[MODE_ANY3] = _any3_records(df, r, p.small_pct / 100.0, p.span_days, keep)
    result[MODE_STRICT] = _strict_records(df, r, p.small_pct / 100.0, p.cum_days, keep)
    return result


def find_drop_after_peak(
    df: pd.DataFrame,
    peak_date,
    peak_high: float,
    *,
    post_peak_days: int = DEFAULT_POST_PEAK_DAYS,
    drop_pct: float = DEFAULT_DROP_PCT / 100.0,
) -> Optional[DropResult]:
    """Check whether min Close within `post_peak_days` trading rows after the
    peak fell at least `drop_pct` (fraction) below the peak High.

    The post-peak window is truncated at the end of available data — the
    fall just has to occur within the rows that exist.
    """
    peak_ts = pd.Timestamp(peak_date)
    post = df[df.index > peak_ts]
    if post.empty:
        return None
    post = post.head(post_peak_days)
    min_close = float(post["Close"].min())
    drop = (peak_high - min_close) / peak_high
    if drop < drop_pct:
        return None
    trough_ts = post["Close"].idxmin()
    return DropResult(
        peak_date=_fmt(peak_ts),
        peak_high=float(peak_high),
        trough_date=_fmt(trough_ts),
        trough_close=min_close,
        drop_pct=round(drop * 100.0, 2),
        days_peak_to_trough=int((trough_ts - peak_ts).days),
        trading_days=int(post.index.get_loc(trough_ts)) + 1,
    )


def screen_droppers(df: pd.DataFrame, p: ScreenParams) -> Optional[DropResult]:
    """Standalone drop screen: peak = first global max High in the period."""
    if len(df) < p.min_bars:
        return None
    peak_ts = df["High"].idxmax()
    return find_drop_after_peak(
        df,
        peak_ts,
        float(df.loc[peak_ts, "High"]),
        post_peak_days=p.post_peak_days,
        drop_pct=p.drop_pct / 100.0,
    )


def run_one_stock(
    area: str,
    ticker: str,
    df: Optional[pd.DataFrame],
    p: ScreenParams,
    *,
    with_drop: bool = False,
) -> StockResult:
    """Run all screens on one stock.

    When with_drop is True, also computes:
      - drop: standalone global-peak drop, and
      - pairs: per mode, the (spike window, drop result) pair with the
        largest drop among that stock's qualifying windows in the mode.
    """
    if df is None or len(df) < p.min_bars:
        return StockResult(area=area, ticker=ticker, skipped_reason="insufficient bars")

    spikes = find_spikes(df, p)

    drop = None
    pairs: dict = {}
    if with_drop:
        drop = screen_droppers(df, p)
        for mode, windows in spikes.items():
            best = None
            for w in windows:
                d = find_drop_after_peak(
                    df,
                    w.peak_date,
                    w.peak_high,
                    post_peak_days=p.post_peak_days,
                    drop_pct=p.drop_pct / 100.0,
                )
                if d is not None and (best is None or d.drop_pct > best[1].drop_pct):
                    best = (w, d)
            if best is not None:
                pairs[mode] = best

    return StockResult(
        area=area, ticker=ticker, spikes=spikes, drop=drop, pairs=pairs
    )


# ── Result assembly ────────────────────────────────────────────────────

def spike_to_dict(sr: StockResult, w: SpikeWindow) -> dict:
    return {
        "ticker": sr.ticker,
        "area": sr.area,
        "mode": w.mode,
        "window_start": w.window_start,
        "window_end": w.window_end,
        "start_open": w.start_open,
        "peak_date": w.peak_date,
        "peak_high": w.peak_high,
        "gain_pct": w.gain_pct,
        "detail": w.detail,
    }


def drop_to_dict(sr: StockResult) -> dict:
    d = sr.drop
    return {
        "ticker": sr.ticker,
        "area": sr.area,
        "peak_date": d.peak_date,
        "peak_high": d.peak_high,
        "trough_date": d.trough_date,
        "trough_close": d.trough_close,
        "drop_pct": d.drop_pct,
        "days_peak_to_trough": d.days_peak_to_trough,
        "trading_days": d.trading_days,
    }


def build_union(stock_results: Sequence[StockResult]) -> list[dict]:
    """Union spike list: one entry per spiking ticker, listing every mode it
    qualified in, with its largest-gain window across those modes."""
    union = []
    for sr in stock_results:
        modes = [m for m in MODES if sr.spikes.get(m)]
        if not modes:
            continue
        best = None
        for m in modes:
            for w in sr.spikes[m]:
                if best is None or w.gain_pct > best.gain_pct:
                    best = w
        entry = spike_to_dict(sr, best)
        entry["modes"] = modes
        union.append(entry)
    union.sort(key=lambda e: e["gain_pct"], reverse=True)
    return union


def _intersection_dict(sr: StockResult, mode: str, qualified: list, w: SpikeWindow, d: DropResult) -> dict:
    return {
        "ticker": sr.ticker,
        "area": sr.area,
        "mode": mode,
        "modes": qualified,
        "window_start": w.window_start,
        "window_end": w.window_end,
        "start_open": w.start_open,
        "peak_date": d.peak_date,
        "peak_high": d.peak_high,
        "gain_pct": w.gain_pct,
        "trough_date": d.trough_date,
        "trough_close": d.trough_close,
        "drop_pct": d.drop_pct,
        "days_peak_to_trough": d.days_peak_to_trough,
    }


def build_intersection(
    stock_results: Sequence[StockResult],
) -> dict:
    """Spike × drop intersection from run_one_stock(..., with_drop=True) results.

    Returns {"per_mode": {mode: [records]}, "union": [records]}. A stock's
    union record is its largest-drop pair across modes; records are sorted
    by drop_pct descending.
    """
    per_mode = {m: [] for m in MODES}
    union_rows = []
    for sr in stock_results:
        if not sr.pairs:
            continue
        qualified = [m for m in MODES if sr.spikes.get(m)]
        for mode, (w, d) in sr.pairs.items():
            per_mode[mode].append(_intersection_dict(sr, mode, qualified, w, d))
        best_mode = max(
            sr.pairs, key=lambda m: (sr.pairs[m][1].drop_pct, sr.pairs[m][0].gain_pct)
        )
        w, d = sr.pairs[best_mode]
        union_rows.append(_intersection_dict(sr, "union", qualified, w, d))

    for m in per_mode:
        per_mode[m].sort(key=lambda r: r["drop_pct"], reverse=True)
    union_rows.sort(key=lambda r: r["drop_pct"], reverse=True)
    return {"per_mode": per_mode, "union": union_rows}
