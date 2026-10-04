"""Pydantic request/response models for the spike & drop screening API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ScreenRequest(BaseModel):
    """Shared POST body for the screening endpoints.

    All fields are optional. Thresholds are in PERCENT units (25.0 == 25%).
    Spike fields are ignored by /api/screen/droppers.
    """

    start_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    exchanges: list[Literal["SS", "SZ"]] = Field(default_factory=lambda: ["SS", "SZ"])
    tickers: Optional[list[str]] = Field(
        None, description="Optional filter; codes with or without .SS/.SZ suffix"
    )
    week_pct: float = Field(25.0, gt=0, description="Week-window spike threshold (%)")
    week_days: int = Field(5, ge=2, description="Trading days in 'a week'")
    cum_days: int = Field(3, ge=2, description="Consecutive days for the cum3 modes")
    small_pct: float = Field(10.0, gt=0, description="Threshold for the cum3 modes (%)")
    span_days: int = Field(10, ge=3, description="2-week span (trading days) for cum3_any3")
    drop_pct: float = Field(15.0, gt=0, description="Post-peak fall threshold (%)")
    post_peak_days: int = Field(10, ge=1, description="Trading days after the peak for the drop")
    per_stock: Literal["max", "first", "all"] = Field(
        "max", description="Which qualifying window to report per (stock, mode)"
    )
    min_bars: int = Field(20, ge=2, description="Minimum rows to screen a stock")
    exclude_b_shares: bool = Field(
        True, description="Exclude B-shares (SS 900xxx USD, SZ 200xxx HKD)"
    )

    @model_validator(mode="after")
    def _check_range(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date")
        return self


class SpikeRecord(BaseModel):
    ticker: str
    area: str
    mode: str
    window_start: str
    window_end: str
    start_open: float
    peak_date: str
    peak_high: float
    gain_pct: float
    detail: dict = Field(default_factory=dict)


class UnionSpikeRecord(SpikeRecord):
    modes: list[str]  # all modes this ticker qualified in


class DropRecord(BaseModel):
    ticker: str
    area: str
    peak_date: str
    peak_high: float
    trough_date: str
    trough_close: float
    drop_pct: float
    days_peak_to_trough: int
    trading_days: int


class IntersectionRecord(BaseModel):
    ticker: str
    area: str
    mode: str          # one of the 4 modes, or "union"
    modes: list[str]   # all modes the ticker qualified in
    window_start: str
    window_end: str
    start_open: float
    peak_date: str
    peak_high: float
    gain_pct: float
    trough_date: str
    trough_close: float
    drop_pct: float
    days_peak_to_trough: int


class ScanStats(BaseModel):
    scanned: int
    skipped: int
    skipped_reasons: dict[str, int]
    requested_tickers: Optional[int]
    matched_tickers: int
    elapsed_sec: float


class GainersResponse(BaseModel):
    params: dict
    stats: ScanStats
    counts: dict[str, int]                    # per mode + "union"
    results: dict[str, list[SpikeRecord]]     # per mode
    union: list[UnionSpikeRecord]


class DroppersResponse(BaseModel):
    params: dict
    stats: ScanStats
    count: int
    results: list[DropRecord]


class IntersectionResponse(BaseModel):
    params: dict
    stats: ScanStats
    counts: dict                              # spike counts per mode, spike union, intersect per mode, intersect union
    intersections: dict[str, list[IntersectionRecord]]  # per mode
    union: list[IntersectionRecord]           # mode = "union"


class ReportResponse(BaseModel):
    files: dict[str, str]                     # {"markdown": ..., "spikes_csv": ..., "intersection_csv": ...}
    report_dir: str
    payload: IntersectionResponse


class TickersResponse(BaseModel):
    areas: dict[str, dict]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
