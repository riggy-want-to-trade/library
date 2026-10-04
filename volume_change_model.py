# ╔══════════════════════════════════════════════════════════════════╗
# ║   3-channel change model: volume driven by a MEASURED input      ║
# ╚══════════════════════════════════════════════════════════════════╝
# Model (u now TIME-VARYING and measured, u(t) = |close return|):
#     dz1/dt = -z1 + a1·u,  dz2/dt = -z2/7 + a2·u,  dz3/dt = -z3/21 + a3·u
#     1 = a1 + a2 + a3,     Q = z1 + z2 + z3   (Q = volume level)
# Observed channels: the volume CHANGE over h = 1, 7, 21 days:
#     y_h(t) = Q(t) - Q(t-h) = sum_i [ z_i(0)·(e^{-t/τi} − e^{-(t-h)/τi})
#                                      + a_i·(F_i(t) − F_i(t-h)) ]
# where F_i is mode i's forced response to the measured u (zero init state).
# Everything is LINEAR in the 5 unknowns (z1(0), z2(0), z3(0), a1, a2 with
# a3 = 1 − a1 − a2), so one joint least squares across all three channels.
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq

TAU = np.array([1.0, 7.0, 21.0])
HORIZONS = (1, 7, 21)
BLUE, ORANGE, BLACK, GRAY = "#2a78d6", "#eb6834", "#0b0b0b", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "axes.titlecolor": "#0b0b0b", "xtick.color": "#898781",
    "ytick.color": "#898781", "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
})

def load(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    close = np.array([float(r[4]) for r in rows])   # column 5 = close
    vol = np.array([float(r[5]) for r in rows])     # column 6 = volume
    return close, vol

def run(ticker):
    close, vol = load(f"America_modified/{ticker}.txt")
    N = len(vol)
    t_all = np.arange(N, dtype=float)
    print(f"\n══════════ {ticker} ══════════  N = {N}")

    # ---- measured input: |daily close return| ------------------------------
    r = np.diff(np.log(close))
    u = np.abs(np.concatenate([[r[0]], r]))
    print(f"u = |close return|: mean {u.mean():.4f}  std {u.std():.4f}")

    # ---- exact forced responses for daily piecewise-constant u -------------
    F = np.zeros((3, N))
    for i in range(3):
        d = np.exp(-1.0 / TAU[i])
        for t in range(1, N):
            F[i, t] = d * F[i, t - 1] + TAU[i] * (1 - d) * u[t]

    H = np.stack([np.exp(-t_all / TAU[i]) for i in range(3)])   # homogeneous modes

    # ---- joint fit across the three change channels -------------------------
    Xs, ys, preds = [], [], []
    for h in HORIZONS:
        idx = np.arange(h, N)
        Hh = np.stack([H[i][idx] - H[i][idx - h] for i in range(3)])
        Fh = np.stack([F[i][idx] - F[i][idx - h] for i in range(3)])
        Xs.append(np.column_stack([Hh[0], Hh[1], Hh[2],
                                   Fh[0] - Fh[2], Fh[1] - Fh[2]]))
        ys.append((vol[idx] - vol[idx - h]) - Fh[2])
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    b, *_ = lstsq(X, y, rcond=None)
    z0 = b[:3]
    a = np.array([b[3], b[4], 1 - b[3] - b[4]])

    print(f"joint 3-channel fit:  z(0) = [{z0[0]:,.0f}, {z0[1]:,.0f}, {z0[2]:,.0f}]")
    print(f"                      a    = [{a[0]:+.4f}, {a[1]:+.4f}, {a[2]:+.4f}]  (sum {a.sum():.4f})")
    for k, h in enumerate(HORIZONS):
        idx = np.arange(h, N)
        yh = vol[idx] - vol[idx - h]
        pred = Xs[k] @ b + (F[2][idx] - F[2][idx - h])
        r2 = 1 - np.sum((yh - pred) ** 2) / np.sum((yh - yh.mean()) ** 2)
        print(f"  channel Δ{h:2d}V: R² = {r2:.4f}")

    # ---- fitted dominance matrix: variance share of each mode per channel ----
    print("  dominance (share of fitted variance per mode per channel):")
    for k, h in enumerate(HORIZONS):
        idx = np.arange(h, N)
        shares = []
        for i in range(3):
            C = z0[i] * (H[i][idx] - H[i][idx - h]) + a[i] * (F[i][idx] - F[i][idx - h])
            shares.append(np.var(C))
        s = np.array(shares) / sum(shares)
        print(f"    Δ{h:2d}V: z1 {s[0]:.2f}  z2 {s[1]:.2f}  z3 {s[2]:.2f}")

    # ---- level cross-check: same model fit directly to the volume LEVEL ----
    Xl = np.column_stack([H[0], H[1], H[2], F[0] - F[2], F[1] - F[2]])
    yl = vol - F[2]
    bl, *_ = lstsq(Xl, yl, rcond=None)
    al = np.array([bl[3], bl[4], 1 - bl[3] - bl[4]])
    predl = Xl @ bl + F[2]
    r2l = 1 - np.sum((vol - predl) ** 2) / np.sum((vol - vol.mean()) ** 2)
    print(f"level fit cross-check:  R² = {r2l:.4f}   a = [{al[0]:+.4f}, {al[1]:+.4f}, {al[2]:+.4f}]")

    # ---- out-of-sample: fit first 900 days, test the rest --------------------
    split = 900
    Xt = np.column_stack([H[0][:split], H[1][:split], H[2][:split],
                          F[0][:split] - F[2][:split], F[1][:split] - F[2][:split]])
    yt = vol[:split] - F[2][:split]
    bt, *_ = lstsq(Xt, yt, rcond=None)
    pred_holdout = Xl[split:] @ bt + F[2][split:]
    r2_hold = 1 - np.sum((vol[split:] - pred_holdout) ** 2) / \
                  np.sum((vol[split:] - vol[split:].mean()) ** 2)
    print(f"out-of-sample (fit 0-900, test 900+): R² = {r2_hold:.4f}")

    # ---- input variants: does the choice of u matter? ------------------------
    for name, uv in (("u = return", np.concatenate([[r[0]], r])),
                     ("u = return²", np.concatenate([[r[0]], r]) ** 2)):
        Fv = np.zeros((3, N))
        for i in range(3):
            d = np.exp(-1.0 / TAU[i])
            for t in range(1, N):
                Fv[i, t] = d * Fv[i, t - 1] + TAU[i] * (1 - d) * uv[t]
        Xv = np.column_stack([H[0], H[1], H[2], Fv[0] - Fv[2], Fv[1] - Fv[2]])
        bv, *_ = lstsq(Xv, vol - Fv[2], rcond=None)
        predv = Xv @ bv + Fv[2]
        r2v = 1 - np.sum((vol - predv) ** 2) / np.sum((vol - vol.mean()) ** 2)
        print(f"level fit with {name}: R² = {r2v:.4f}")

    # ---- plots ----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.plot(t_all, vol, lw=0.8, color=GRAY, label="volume")
    ax1.plot(t_all, predl, lw=1.5, color=BLUE, label="model (measured u = |return|)")
    ax1.set_xlabel("trading days"); ax1.set_ylabel("volume")
    ax1.set_title(f"{ticker} — level: model vs data (R² = {r2l:.3f})")
    ax1.legend(fontsize=9)
    h = 7
    idx = np.arange(h, N)
    yh = vol[idx] - vol[idx - h]
    predh = Xs[HORIZONS.index(h)] @ b + (F[2][idx] - F[2][idx - h])
    ax2.plot(idx[:250], yh[:250], lw=0.8, color=GRAY, label="Δ7V measured")
    ax2.plot(idx[:250], predh[:250], lw=1.5, color=ORANGE, label="Δ7V model")
    ax2.set_xlabel("trading days"); ax2.set_ylabel("7-day volume change")
    ax2.set_title(f"{ticker} — 7-day change channel (first 250 days)")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{ticker.lower()}_change_model.png", dpi=100)
    print(f"plot saved: {ticker.lower()}_change_model.png")
    return dict(a=a, z0=z0, r2_level=r2l, r2_holdout=r2_hold)

run("MSFT")
run("AAPL")
