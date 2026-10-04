"""FastAPI spike & drop screening service.

Endpoints:
    GET  /health                       — service status and data areas
    GET  /api/tickers?exchanges=SS,SZ  — available tickers per area
    POST /api/screen/gainers           — spike lists per mode + union
    POST /api/screen/droppers          — standalone post-peak drop list
    POST /api/screen/intersection      — spike × drop intersection
    POST /api/report                   — intersection + report files
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from .config import DATA_DIR, HOST, MAX_WORKERS, MODES, PORT, REPORT_DIR
from .loader import (
    DataNotFoundError,
    NoDataInRangeError,
    count_tickers,
    discover_tickers,
    load_all,
)
from .models import (
    DroppersResponse,
    ErrorResponse,
    GainersResponse,
    IntersectionResponse,
    ReportResponse,
    ScreenRequest,
    TickersResponse,
)
from .report import write_report
from .screener import (
    ScreenParams,
    build_intersection,
    build_union,
    drop_to_dict,
    run_one_stock,
    spike_to_dict,
)

# ── App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="A-Share Spike & Drop Screener",
    version="1.0.0",
    description=(
        "Screens stock_data/SS and stock_data/SZ daily OHLCV for stocks that "
        "spiked (week-window threshold or one of three 3-day patterns) and "
        "then fell after their peak."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

VALID_AREAS = ("SS", "SZ")


# ── Exception handlers ─────────────────────────────────────────────────

@app.exception_handler(DataNotFoundError)
async def data_not_found_handler(request, exc: DataNotFoundError):
    return JSONResponse(status_code=404, content={"error": "Ticker not found", "detail": str(exc)})


@app.exception_handler(NoDataInRangeError)
async def no_data_in_range_handler(request, exc: NoDataInRangeError):
    return JSONResponse(status_code=404, content={"error": "No data in range", "detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


# ── Helpers ────────────────────────────────────────────────────────────

def _params_from(req: ScreenRequest) -> ScreenParams:
    return ScreenParams(
        week_pct=req.week_pct,
        week_days=req.week_days,
        cum_days=req.cum_days,
        small_pct=req.small_pct,
        span_days=req.span_days,
        drop_pct=req.drop_pct,
        post_peak_days=req.post_peak_days,
        per_stock=req.per_stock,
        min_bars=req.min_bars,
    )


def _validate_exchanges(exchanges: list[str]) -> None:
    unknown = [a for a in exchanges if a not in VALID_AREAS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown exchange(s): {', '.join(unknown)}. Available: {', '.join(VALID_AREAS)}",
        )


def _scan(req: ScreenRequest, p: ScreenParams, with_drop: bool):
    """Discover + load + screen all requested tickers. Returns (results, stats)."""
    t0 = time.perf_counter()
    _validate_exchanges(req.exchanges)

    ticker_list = [t.strip().upper() for t in req.tickers if t.strip()] if req.tickers else None
    bundle = load_all(
        req.exchanges,
        req.start_date,
        req.end_date,
        ticker_list,
        max_workers=MAX_WORKERS,
        exclude_b_shares=req.exclude_b_shares,
    )

    stock_results = [
        run_one_stock(area, ticker, df, p, with_drop=with_drop)
        for area, ticker, df in bundle.frames
    ]
    elapsed = time.perf_counter() - t0

    reasons = Counter(s["reason"] for s in bundle.skipped)
    for sr in stock_results:
        if sr.skipped_reason:
            reasons[sr.skipped_reason] += 1
    matched = sum(1 for sr in stock_results if not sr.skipped_reason)
    stats = {
        "scanned": len(bundle.frames) + len(bundle.skipped),
        "skipped": sum(reasons.values()),
        "skipped_reasons": dict(reasons),
        "requested_tickers": len(ticker_list) if ticker_list else None,
        "matched_tickers": matched,
        "elapsed_sec": round(elapsed, 2),
    }
    return stock_results, stats


def _spike_lists(stock_results) -> tuple[dict, list[dict]]:
    """Per-mode spike lists (sorted by gain desc) and the union list."""
    per_mode = {m: [] for m in MODES}
    for sr in stock_results:
        for m in MODES:
            for w in sr.spikes.get(m, []):
                per_mode[m].append(spike_to_dict(sr, w))
    for m in per_mode:
        per_mode[m].sort(key=lambda r: r["gain_pct"], reverse=True)
    union = build_union(stock_results)
    return per_mode, union


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Service status and available data areas."""
    areas = {area: count_tickers(area) for area in VALID_AREAS}
    return {"status": "ok", "data_dir": DATA_DIR, "areas": areas}


@app.get("/api/tickers", response_model=TickersResponse)
async def list_tickers(
    exchanges: Optional[str] = Query(
        default=None, description="Comma-separated areas, e.g. SS,SZ. Defaults to both."
    ),
):
    """List available tickers per market area."""
    wanted = [a.strip() for a in exchanges.split(",")] if exchanges else list(VALID_AREAS)
    _validate_exchanges(wanted)
    out = {}
    for area in wanted:
        tickers = discover_tickers(area)
        out[area] = {"count": len(tickers), "tickers": tickers}
    return TickersResponse(areas=out)


@app.post("/api/screen/gainers", response_model=GainersResponse)
async def screen_gainers(body: ScreenRequest):
    """Screen for spiking stocks — one list per mode plus the union list.

    Modes: week_pct25 (max High vs window-start Open over week_days trading
    rows), cum3_consecutive (3 consecutive rows, max High vs first Open),
    cum3_any3 (top-3 day gains within a span_days span), cum3_strict
    (3 consecutive rows each gaining small_pct intraday).
    """
    p = _params_from(body)
    stock_results, stats = _scan(body, p, with_drop=False)
    per_mode, union = _spike_lists(stock_results)
    counts = {f"spike_{m}": len(per_mode[m]) for m in MODES}
    counts["spike_union"] = len(union)
    return GainersResponse(
        params=body.model_dump(),
        stats=stats,
        counts=counts,
        results=per_mode,
        union=union,
    )


@app.post("/api/screen/droppers", response_model=DroppersResponse)
async def screen_droppers(body: ScreenRequest):
    """Standalone drop screen over all scanned stocks.

    Peak = the stock's highest High in the date range; qualifies when the
    min Close within post_peak_days trading rows after the peak fell at
    least drop_pct below the peak High.
    """
    p = _params_from(body)
    stock_results, stats = _scan(body, p, with_drop=True)
    drops = [drop_to_dict(sr) for sr in stock_results if sr.drop is not None]
    drops.sort(key=lambda r: r["drop_pct"], reverse=True)
    return DroppersResponse(
        params=body.model_dump(),
        stats=stats,
        count=len(drops),
        results=drops,
    )


@app.post("/api/screen/intersection", response_model=IntersectionResponse)
async def screen_intersection(body: ScreenRequest):
    """Spike × drop intersection.

    For each spiking stock, per mode, checks whether the min Close within
    post_peak_days trading rows after the spike window's peak fell at least
    drop_pct below that peak. Returns per-mode lists and a union list.
    """
    p = _params_from(body)
    stock_results, stats = _scan(body, p, with_drop=True)
    per_mode, union = _spike_lists(stock_results)
    inter = build_intersection(stock_results)

    counts = {f"spike_{m}": len(per_mode[m]) for m in MODES}
    counts["spike_union"] = len(union)
    counts.update({f"intersect_{m}": len(inter["per_mode"][m]) for m in MODES})
    counts["intersect_union"] = len(inter["union"])

    return IntersectionResponse(
        params=body.model_dump(),
        stats=stats,
        counts=counts,
        intersections=inter["per_mode"],
        union=inter["union"],
    )


@app.post("/api/report", response_model=ReportResponse)
async def generate_report(body: ScreenRequest):
    """Run the full intersection screen and write a report to disk.

    Writes timestamped markdown + CSV files under the reports directory
    and returns their paths together with the intersection payload.
    """
    p = _params_from(body)
    stock_results, stats = _scan(body, p, with_drop=True)
    per_mode, union = _spike_lists(stock_results)
    inter = build_intersection(stock_results)

    counts = {f"spike_{m}": len(per_mode[m]) for m in MODES}
    counts["spike_union"] = len(union)
    counts.update({f"intersect_{m}": len(inter["per_mode"][m]) for m in MODES})
    counts["intersect_union"] = len(inter["union"])

    payload = {
        "params": body.model_dump(),
        "stats": stats,
        "counts": counts,
        "spikes": per_mode,
        "spike_union": union,
        "intersections": inter["per_mode"],
        "union": inter["union"],
    }
    files = write_report(payload)

    intersection_response = IntersectionResponse(
        params=body.model_dump(),
        stats=stats,
        counts=counts,
        intersections=inter["per_mode"],
        union=inter["union"],
    )
    return ReportResponse(
        files=files,
        report_dir=str(REPORT_DIR),
        payload=intersection_response,
    )


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print(f"Data directory: {DATA_DIR}")
    print(f"Report directory: {REPORT_DIR}")
    uvicorn.run(app, host=HOST, port=PORT)
