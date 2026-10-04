# spike_screen — A-Share Spike & Drop Screener

FastAPI REST service that screens the daily OHLCV files in
`stock_data/SS` (Shanghai) and `stock_data/SZ` (Shenzhen) for stocks that
**spiked**, then **fell after their peak**.

## Quick start

```bash
cd D:\rain\library
.venv/Scripts/python.exe -m spike_screen.main
# -> http://127.0.0.1:8001/docs  (interactive API docs)
```

## Endpoints

| Method / Path | Description |
|---|---|
| `GET /health` | Service status, data dir, ticker counts per area |
| `GET /api/tickers?exchanges=SS,SZ` | Available tickers per area |
| `POST /api/screen/gainers` | Spike lists — one per mode + a union list |
| `POST /api/screen/droppers` | Standalone drop list over all scanned stocks |
| `POST /api/screen/intersection` | Spike × drop intersection (per mode + union) |
| `POST /api/report` | Same as intersection + writes markdown/CSV files to `reports/` |

### Spike modes (each computed as its own list)

| Mode | Rule (all thresholds in percent units) |
|---|---|
| `week_pct25` | Rolling `week_days` (5) trading days: (max High − Open of window's first day) ÷ Open ≥ `week_pct` (25) |
| `cum3_consecutive` | Any `cum_days` (3) consecutive trading days: (max High − Open of day 1) ÷ Open ≥ `small_pct` (10) |
| `cum3_any3` | Any 3 days inside a `span_days` (10) trading-day span: sum of the days' (High − Open) ÷ Open ≥ `small_pct` (10) |
| `cum3_strict` | `cum_days` (3) consecutive days where EACH day has (High − Open) ÷ Open ≥ `small_pct` (10) |

### Drop rule

After the peak (max High of the qualifying spike window), the minimum
Close within the next `post_peak_days` (10) trading rows must fall ≥
`drop_pct` (15) below the peak High. The standalone `/api/screen/droppers`
uses each stock's global max High in the date range as the peak.

### Request parameters (all optional)

```json
{
  "start_date": "2021-09-24",   // YYYY-MM-DD; defaults to 5 years back
  "end_date": "2026-09-24",     // defaults to today
  "exchanges": ["SS", "SZ"],
  "tickers": ["600000.SS"],     // optional filter; codes with or without suffix
  "week_pct": 25.0,             // week-window spike threshold (%)
  "week_days": 5,               // trading days in "a week"
  "cum_days": 3,                // consecutive days for the cum3 modes
  "small_pct": 10.0,            // cum3 modes threshold (%)
  "span_days": 10,              // 2-week span (trading days) for cum3_any3
  "drop_pct": 15.0,             // post-peak fall threshold (%)
  "post_peak_days": 10,         // trading days after the peak for the drop
  "per_stock": "max",           // "max" | "first" | "all" — which qualifying window to report
  "min_bars": 20,               // minimum rows to screen a stock
  "exclude_b_shares": true      // exclude SS 900xxx (USD) / SZ 200xxx (HKD)
}
```

Example:

```bash
curl -X POST http://127.0.0.1:8001/api/report \
  -H "Content-Type: application/json" \
  -d '{"week_pct": 25.0, "small_pct": 10.0, "drop_pct": 15.0}'
```

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `SPIKE_DATA_DIR` | `<repo>/stock_data` | Data root (contains `SS/`, `SZ/`) |
| `SPIKE_REPORT_DIR` | `<repo>/reports` | Report output directory |
| `SPIKE_HOST` / `SPIKE_PORT` | `0.0.0.0` / `8001` | Server bind (backtest_server uses 8000) |
| `SPIKE_MAX_WORKERS` | `8` | Parallel CSV readers |

## Algorithm notes

- **All windows roll over rows (trading days), never calendar days** — this
  correctly handles A-share suspension gaps: a "week" is 5 trading rows,
  which may span more calendar time for a suspended stock.
- Prices in the data files are dividend/split back-adjusted, so percentage
  gains computed here are true returns.
- **Discontinuity guard**: when a day's Close jumps >2× (or <0.5×) vs the
  previous Close, the series is split into segments (bad feed rows, e.g.
  300276.SZ's ×100 artifact row on 2025-04-25, and legit suspension
  resumptions / IPO listing days). Spike windows never cross a segment
  boundary, so fake cross-regime gains are excluded while real IPO-day
  spikes within one continuous segment still qualify.
- The post-peak drop window uses up to `post_peak_days` trading rows after
  the peak date and truncates at the end of available data — the fall just
  has to occur within the rows that exist.
- Data files are `<code>.SS.txt` / `<code>.SZ.txt`, no header, columns
  `Date, Open, High, Low, Close, Volume` (note: `backtest_server` previously
  had Open/Close swapped — fixed).
- `per_stock="max"` (default) keeps one window per (stock, mode) — the
  largest gain — to bound response sizes; use `"all"` to get every
  qualifying window.
