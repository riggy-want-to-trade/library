"""Report writer — markdown + CSV files under the reports directory."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import MODES, REPORT_DIR

SPIKE_CSV_COLS = [
    "ticker", "area", "mode", "window_start", "window_end", "start_open",
    "peak_date", "peak_high", "gain_pct",
]
INTERSECTION_CSV_COLS = [
    "ticker", "area", "mode", "modes", "window_start", "window_end",
    "start_open", "gain_pct", "peak_date", "peak_high",
    "trough_date", "trough_close", "drop_pct", "days_peak_to_trough",
]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _union_table_rows(rows: list[dict]) -> list[list]:
    out = []
    for i, r in enumerate(rows, 1):
        out.append([
            i, r["ticker"], r["area"],
            "; ".join(r.get("modes", [r["mode"]])) if r.get("mode") == "union" else r["mode"],
            f"{r['window_start']} → {r['window_end']}",
            f"{r['gain_pct']:.2f}%",
            r["peak_date"], f"{r['peak_high']:.2f}",
            r["trough_date"], f"{r['trough_close']:.2f}",
            f"{r['drop_pct']:.2f}%", r["days_peak_to_trough"],
        ])
    return out


def build_markdown(payload: dict) -> str:
    """Build the full markdown report from the combined payload dict."""
    params = payload["params"]
    stats = payload["stats"]
    counts = payload["counts"]

    lines = [
        "# A-Share Spike & Drop Screen Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Parameters",
        "",
        _md_table(["Parameter", "Value"], [[k, v] for k, v in params.items()]),
        "",
        "## Scan stats",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Tickers scanned", stats["scanned"]],
                ["Matched (screened)", stats["matched_tickers"]],
                ["Skipped", stats["skipped"]],
                ["Skipped reasons", ", ".join(f"{k}: {v}" for k, v in stats["skipped_reasons"].items()) or "-"],
                ["Elapsed (s)", f"{stats['elapsed_sec']:.2f}"],
            ],
        ),
        "",
        "## Spike counts",
        "",
        _md_table(
            ["Mode", "Count"],
            [[m, counts.get(f"spike_{m}", 0)] for m in MODES]
            + [["union", counts.get("spike_union", 0)]],
        ),
        "",
        "## Intersection summary",
        "",
        _md_table(
            ["Mode", "Spike count", "Intersection count"],
            [[m, counts.get(f"spike_{m}", 0), counts.get(f"intersect_{m}", 0)] for m in MODES]
            + [["union", counts.get("spike_union", 0), counts.get("intersect_union", 0)]],
        ),
        "",
        "## Intersection — Union",
        "",
        "Stocks that spiked AND fell >= the drop threshold after their spike peak.",
        "",
    ]

    headers = [
        "#", "Ticker", "Area", "Mode(s)", "Spike window", "Gain %",
        "Peak date", "Peak high", "Trough date", "Trough close",
        "Drop %", "Days peak→trough",
    ]
    union_rows = payload.get("union", [])
    if union_rows:
        lines += [_md_table(headers, _union_table_rows(union_rows))]
    else:
        lines += ["_No stocks matched the intersection criteria._"]
    lines += [""]

    per_mode = payload.get("intersections", {})
    for mode in MODES:
        lines += [f"## Intersection — {mode}", ""]
        rows = per_mode.get(mode, [])
        if rows:
            lines += [_md_table(headers, _union_table_rows(rows))]
        else:
            lines += ["_No matches._"]
        lines += [""]

    return "\n".join(lines)


def build_spikes_df(payload: dict) -> pd.DataFrame:
    """All spike records (per mode + union) as one DataFrame."""
    rows = []
    for mode in MODES:
        for r in payload.get("spikes", {}).get(mode, []):
            row = {k: r.get(k) for k in SPIKE_CSV_COLS}
            row["mode"] = mode
            rows.append(row)
    for r in payload.get("spike_union", []):
        row = {k: r.get(k) for k in SPIKE_CSV_COLS}
        row["mode"] = "union"
        rows.append(row)
    return pd.DataFrame(rows, columns=SPIKE_CSV_COLS)


def build_intersections_df(payload: dict) -> pd.DataFrame:
    """All intersection records (per mode + union) as one DataFrame."""
    rows = []
    for mode in MODES:
        for r in payload.get("intersections", {}).get(mode, []):
            row = {k: r.get(k) for k in INTERSECTION_CSV_COLS}
            rows.append(row)
    for r in payload.get("union", []):
        row = {k: r.get(k) for k in INTERSECTION_CSV_COLS}
        rows.append(row)
    df = pd.DataFrame(rows, columns=INTERSECTION_CSV_COLS)
    if not df.empty:
        # modes column as a readable string
        df["modes"] = df["modes"].apply(lambda m: "; ".join(m) if isinstance(m, list) else m)
    return df


def write_report(payload: dict, out_dir: Path | None = None) -> dict:
    """Write the markdown + CSV report files, returning their paths.

    Files (timestamped) under `out_dir` (default REPORT_DIR):
        screen_report_<ts>.md
        screen_spikes_<ts>.csv
        screen_intersections_<ts>.csv
    """
    out_dir = Path(out_dir) if out_dir else REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()

    md_path = out_dir / f"screen_report_{ts}.md"
    spikes_path = out_dir / f"screen_spikes_{ts}.csv"
    inter_path = out_dir / f"screen_intersections_{ts}.csv"

    md_path.write_text(build_markdown(payload), encoding="utf-8")
    build_spikes_df(payload).to_csv(spikes_path, index=False, encoding="utf-8-sig")
    build_intersections_df(payload).to_csv(inter_path, index=False, encoding="utf-8-sig")

    return {
        "markdown": str(md_path),
        "spikes_csv": str(spikes_path),
        "intersection_csv": str(inter_path),
    }
