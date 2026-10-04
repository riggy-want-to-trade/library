# Volume Change Model — 3-Channel System ID with a Measured Input

Replication guide for the three-horizon change model fitted to MSFT and AAPL
volume with a **measured, time-varying input**. Every code block is
self-contained and copy-paste runnable in order — no `.py` file needed.

---

## The model

```
dz1/dt = -z1 + a1·u(t),   dz2/dt = -z2/7 + a2·u(t),   dz3/dt = -z3/21 + a3·u(t)
1 = a1 + a2 + a3,         Q = z1 + z2 + z3           (Q = volume level)
```

`u(t)` is now **measured** (not a constant): `u = |daily close return|`.
The observed channels are the volume **change** over h = 1, 7, 21 days:

```
y_h(t) = Q(t) − Q(t−h) = Σ_i [ z_i(0)·(e^(−t/τi) − e^(−(t−h)/τi))
                              + a_i·(F_i(t) − F_i(t−h)) ]
```

where `F_i` is mode i's forced response to the measured `u` (zero initial
state). Everything is linear in the 5 unknowns `(z1(0), z2(0), z3(0), a1, a2)`
with `a3 = 1 − a1 − a2`, so one joint least-squares fit stacks all three
channels.

## Setup

```bash
pip install numpy matplotlib
```

Data: `America_modified/MSFT.txt` / `AAPL.txt` — column 5 = close, column 6 =
volume, 1,255 daily rows each, 2021-08-16 .. 2026-08-14.

---

## Step 1 — load, input, and forced responses

```python
import csv

import matplotlib
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

def build(ticker):
    close, vol = load(f"America_modified/{ticker}.txt")
    N = len(vol)
    r = np.diff(np.log(close))
    u = np.abs(np.concatenate([[r[0]], r]))         # measured input: |return|
    F = np.zeros((3, N))                            # exact forced responses
    for i in range(3):
        d = np.exp(-1.0 / TAU[i])
        for t in range(1, N):                       # daily piecewise-constant u
            F[i, t] = d * F[i, t - 1] + TAU[i] * (1 - d) * u[t]
    H = np.stack([np.exp(-np.arange(N, dtype=float) / TAU[i]) for i in range(3)])
    Xs, ys = [], []
    for h in HORIZONS:                              # design per change channel
        idx = np.arange(h, N)
        Hh = np.stack([H[i][idx] - H[i][idx - h] for i in range(3)])
        Fh = np.stack([F[i][idx] - F[i][idx - h] for i in range(3)])
        Xs.append(np.column_stack([Hh[0], Hh[1], Hh[2],
                                   Fh[0] - Fh[2], Fh[1] - Fh[2]]))
        ys.append((vol[idx] - vol[idx - h]) - Fh[2])
    return dict(close=close, vol=vol, N=N, u=u, F=F, H=H,
                Xs=Xs, ys=ys, X=np.vstack(Xs), y=np.concatenate(ys))
```

---

## Step 2 — the joint 3-channel fit (and diagnostics)

```python
def run(ticker):
    D = build(ticker)
    vol, N, u, F, H, Xs, ys = (D[k] for k in
                               ("vol", "N", "u", "F", "H", "Xs", "ys"))
    X, y = D["X"], D["y"]
    print(f"\n========== {ticker} ==========  N = {N}")
    print(f"u = |close return|: mean {u.mean():.4f}  std {u.std():.4f}")

    b, *_ = lstsq(X, y, rcond=None)                 # the one joint fit
    z0 = b[:3]
    a = np.array([b[3], b[4], 1 - b[3] - b[4]])
    print(f"joint 3-channel fit:  z(0) = [{z0[0]:,.0f}, {z0[1]:,.0f}, {z0[2]:,.0f}]")
    print(f"                      a    = [{a[0]:+.4f}, {a[1]:+.4f}, {a[2]:+.4f}]  "
          f"(sum {a.sum():.4f})")
    for k, h in enumerate(HORIZONS):
        idx = np.arange(h, N)
        yh = vol[idx] - vol[idx - h]
        pred = Xs[k] @ b + (F[2][idx] - F[2][idx - h])
        r2 = 1 - np.sum((yh - pred) ** 2) / np.sum((yh - yh.mean()) ** 2)
        print(f"  channel d{h:2d}V: R^2 = {r2:.4f}")

    print("  dominance (share of fitted variance per mode per channel):")
    for k, h in enumerate(HORIZONS):
        idx = np.arange(h, N)
        shares = []
        for i in range(3):
            C = z0[i] * (H[i][idx] - H[i][idx - h]) \
                + a[i] * (F[i][idx] - F[i][idx - h])
            shares.append(np.var(C))
        s = np.array(shares) / sum(shares)
        print(f"    d{h:2d}V: z1 {s[0]:.2f}  z2 {s[1]:.2f}  z3 {s[2]:.2f}")

    # level cross-check: same unknowns, fit the volume LEVEL directly
    Xl = np.column_stack([H[0], H[1], H[2], F[0] - F[2], F[1] - F[2]])
    yl = vol - F[2]
    bl, *_ = lstsq(Xl, yl, rcond=None)
    al = np.array([bl[3], bl[4], 1 - bl[3] - bl[4]])
    predl = Xl @ bl + F[2]
    r2l = 1 - np.sum((vol - predl) ** 2) / np.sum((vol - vol.mean()) ** 2)
    print(f"level fit cross-check:  R^2 = {r2l:.4f}   "
          f"a = [{al[0]:+.4f}, {al[1]:+.4f}, {al[2]:+.4f}]")

    # out-of-sample: fit first 900 days, test the rest
    split = 900
    Xt = np.column_stack([H[0][:split], H[1][:split], H[2][:split],
                          F[0][:split] - F[2][:split], F[1][:split] - F[2][:split]])
    bt, *_ = lstsq(Xt, vol[:split] - F[2][:split], rcond=None)
    pred_hold = Xl[split:] @ bt + F[2][split:]
    r2h = 1 - np.sum((vol[split:] - pred_hold) ** 2) / \
              np.sum((vol[split:] - vol[split:].mean()) ** 2)
    print(f"out-of-sample (fit 0-900, test 900+): R^2 = {r2h:.4f}")

    # input variants: is |return| the right u?
    r = np.diff(np.log(D["close"]))
    for name, uv in (("u = return", np.concatenate([[r[0]], r])),
                     ("u = return^2", np.concatenate([[r[0]], r]) ** 2)):
        Fv = np.zeros((3, N))
        for i in range(3):
            d = np.exp(-1.0 / TAU[i])
            for t in range(1, N):
                Fv[i, t] = d * Fv[i, t - 1] + TAU[i] * (1 - d) * uv[t]
        Xv = np.column_stack([H[0], H[1], H[2], Fv[0] - Fv[2], Fv[1] - Fv[2]])
        bv, *_ = lstsq(Xv, vol - Fv[2], rcond=None)
        predv = Xv @ bv + Fv[2]
        r2v = 1 - np.sum((vol - predv) ** 2) / np.sum((vol - vol.mean()) ** 2)
        print(f"level fit with {name}: R^2 = {r2v:.4f}")

    # plots: level fit + 7-day change channel
    t_all = np.arange(N, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.plot(t_all, vol, lw=0.8, color=GRAY, label="volume")
    ax1.plot(t_all, predl, lw=1.5, color=BLUE, label="model (measured u = |return|)")
    ax1.set_xlabel("trading days"); ax1.set_ylabel("volume")
    ax1.set_title(f"{ticker} — level: model vs data (R^2 = {r2l:.3f})")
    ax1.legend(fontsize=9)
    h = 7
    idx = np.arange(h, N)
    yh = vol[idx] - vol[idx - h]
    predh = Xs[HORIZONS.index(h)] @ b + (F[2][idx] - F[2][idx - h])
    ax2.plot(idx[:250], yh[:250], lw=0.8, color=GRAY, label="d7V measured")
    ax2.plot(idx[:250], predh[:250], lw=1.5, color=ORANGE, label="d7V model")
    ax2.set_xlabel("trading days"); ax2.set_ylabel("7-day volume change")
    ax2.set_title(f"{ticker} — 7-day change channel (first 250 days)")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    plt.show()      # or fig.savefig(f"{ticker.lower()}_change_model.png", dpi=100)
    return D

RES_MSFT = run("MSFT")
RES_AAPL = run("AAPL")
```

Expected output:

```
========== MSFT ==========  N = 1255
u = |close return|: mean 0.0125  std 0.0124
joint 3-channel fit:  z(0) = [-4,912,802, 30,889,887, -38,410,115]
                      a    = [-235039744.6224, +394344588.8362, -159304843.2138]  (sum 1.0000)
  channel d 1V: R^2 = 0.0139
  channel d 7V: R^2 = 0.0911
  channel d21V: R^2 = 0.1243
  dominance (share of fitted variance per mode per channel):
    d 1V: z1 0.16  z2 0.72  z3 0.12
    d 7V: z1 0.05  z2 0.77  z3 0.18
    d21V: z1 0.03  z2 0.71  z3 0.26
level fit cross-check:  R^2 = 0.1106   a = [-231670769.8880, +183961022.7692, +47709748.1188]
out-of-sample (fit 0-900, test 900+): R^2 = 0.1415
level fit with u = return: R^2 = -4.5386
level fit with u = return^2: R^2 = -0.5223

========== AAPL ==========  N = 1255
u = |close return|: mean 0.0125  std 0.0124
joint 3-channel fit:  z(0) = [58,181,290, 26,940,250, -76,742,148]
                      a    = [-535858478.8605, +934554292.1501, -398695812.2896]  (sum 1.0000)
  channel d 1V: R^2 = 0.0216
  channel d 7V: R^2 = 0.1173
  channel d21V: R^2 = 0.1651
  dominance (share of fitted variance per mode per channel):
    d 1V: z1 0.15  z2 0.71  z3 0.14
    d 7V: z1 0.05  z2 0.75  z3 0.21
    d21V: z1 0.02  z2 0.67  z3 0.31
level fit cross-check:  R^2 = 0.0443   a = [-456299674.3942, +307358333.5575, +148941341.8367]
out-of-sample (fit 0-900, test 900+): R^2 = -1.1098
level fit with u = return: R^2 = -4.4780
level fit with u = return^2: R^2 = -1.1674
```

---

## Step 3 — can the huge a's be stabilised? (ridge and non-negativity)

The ±10⁸ cancelling a's suggest collinear columns. Two cures, both attempted:

```python
# ---- ridge on standardised columns ----
def ridge_grid(ticker, D, lams=(0, 1, 10, 100, 1000)):
    X, y = D["X"], D["y"]
    Xm, Xs_ = X.mean(axis=0), X.std(axis=0)
    Xstd = (X - Xm) / Xs_
    print(f"--- {ticker}: joint 3-channel fit, ridge on standardised columns ---")
    for lam in lams:
        Xa = np.vstack([Xstd, np.sqrt(lam) * np.eye(X.shape[1])])
        ya = np.concatenate([y, np.zeros(X.shape[1])])
        b, *_ = lstsq(Xa, ya, rcond=None)
        b = b / Xs_                                 # un-standardise
        a = np.array([b[3], b[4], 1 - b[3] - b[4]])
        pred = X @ b
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"  lam={lam:5d}: R^2 = {r2:.4f}   "
              f"a = [{a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}]   "
              f"z0 = [{b[0]:,.0f}, {b[1]:,.0f}, {b[2]:,.0f}]")

# ---- constrained: a_i >= 0, sum = 1 (active-set enumeration, numpy only) ----
def solve_constrained(X, y):
    Hc, A1, A2 = X[:, :3], X[:, 3], X[:, 4]
    cands = []
    def add(name, Xr, yr, fill):
        br, *_ = lstsq(Xr, yr, rcond=None)
        res = float((Xr @ br - yr) @ (Xr @ br - yr))
        b = np.zeros(5); b[:len(br)] = br
        if fill is not None:
            for idx, val in fill.items(): b[idx] = val
        cands.append((res, name, b))
    add("interior", X, y, None)
    add("a1=0", np.column_stack([Hc, A2]), y, {3: 0.0})
    add("a2=0", np.column_stack([Hc, A1]), y, {4: 0.0})
    add("a1+a2=1", np.column_stack([Hc, A1 - A2]), y - A2, None)
    add("a1=0,a2=0", Hc, y, {3: 0.0, 4: 0.0})
    add("a1=0,a2=1", Hc, y - A2, {3: 0.0, 4: 1.0})
    add("a2=0,a1=1", Hc, y - A1, {3: 1.0, 4: 0.0})
    best = None
    for res, name, b in cands:
        a1, a2 = b[3], b[4]
        if name == "a1+a2=1":
            a2 = 1 - a1
            if 0 <= a1 <= 1:
                b[4] = a2
            else:
                continue
        if a1 >= -1e-9 and a2 >= -1e-9 and a1 + a2 <= 1 + 1e-9:
            if best is None or res < best[0]:
                best = (res, name, b.copy())
    res, name, b = best
    return b, res, name

for ticker, D in (("MSFT", RES_MSFT), ("AAPL", RES_AAPL)):
    ridge_grid(ticker, D)
    b, res, name = solve_constrained(D["X"], D["y"])
    a = np.array([b[3], b[4], 1 - b[3] - b[4]])
    r2 = 1 - res / np.sum((D["y"] - D["y"].mean()) ** 2)
    print(f"{ticker}: constrained (a>=0, sum=1) [active: {name}]: R^2 = {r2:.4f}")
    print(f"   a = [{a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}]   "
          f"z0 = [{b[0]:,.0f}, {b[1]:,.0f}, {b[2]:,.0f}]")
```

Expected output:

```
--- MSFT: joint 3-channel fit, ridge on standardised columns ---
  lam=    0: R^2 = 0.0870   a = [-235079255.484, +394743559.474, -159664302.991]   z0 = [-4,741,135, 30,452,519, -37,732,416]
  lam=    1: R^2 = 0.0870   a = [-234686703.872, +394000826.890, -159314122.019]   z0 = [-4,690,910, 30,311,080, -37,593,644]
  lam=   10: R^2 = 0.0870   a = [-231215933.538, +387438138.880, -156222204.342]   z0 = [-4,258,145, 29,082,944, -36,385,005]
  lam=  100: R^2 = 0.0851   a = [-201703482.540, +331932884.861, -130229401.321]   z0 = [-1,323,439, 20,121,663, -27,322,444]
  lam= 1000: R^2 = 0.0528   a = [-92142072.069, +131337498.274, -39195425.205]   z0 = [1,646,099, 3,008,268, -7,228,080]
MSFT: constrained (a>=0, sum=1) [active: a1=0]: R^2 = 0.0017
   a = [0.0000, 0.0000, 1.0000]   z0 = [-4,933,934, 17,283,460, -24,267,895]
--- AAPL: joint 3-channel fit, ridge on standardised columns ---
  lam=    0: R^2 = 0.1159   a = [-535843006.899, +934339979.318, -398496971.418]   z0 = [57,852,455, 27,801,453, -78,066,477]
  lam=    1: R^2 = 0.1159   a = [-534747788.600, +932385164.993, -397637375.393]   z0 = [57,896,615, 27,539,502, -77,766,357]
  lam=   10: R^2 = 0.1158   a = [-525094171.841, +915157812.960, -390063640.119]   z0 = [58,257,771, 25,281,143, -75,160,281]
  lam=  100: R^2 = 0.1125   a = [-445121842.379, +772643293.638, -327521450.259]   z0 = [59,398,523, 9,852,724, -56,103,444]
  lam= 1000: R^2 = 0.0641   a = [-178806716.626, +300665425.548, -121858707.923]   z0 = [41,560,226, -8,045,282, -17,100,079]
AAPL: constrained (a>=0, sum=1) [active: a1=0]: R^2 = 0.0081
   a = [0.0000, 0.0000, 1.0000]   z0 = [66,165,495, -33,991,930, -15,782,532]
```

---

## Results summary

| | MSFT | AAPL |
|---|---|---|
| Level R², constant-u model (previous doc) | 0.003 | 0.018 |
| Level R², measured-u model | **0.111** | **0.044** |
| Channel R²: Δ₁V / Δ₇V / Δ₂₁V | 0.014 / 0.091 / **0.124** | 0.022 / 0.117 / **0.165** |
| Out-of-sample R² (fit 0–900, test 900+) | +0.14 | −1.11 |
| Input sanity: u = return / return² | −4.5 / −0.5 | −4.5 / −1.2 |
| a from plain LS | ±10⁸ cancelling | ±10⁸ cancelling |
| a with aᵢ ≥ 0 constraint | [0, 0, 1], R² 0.002 | [0, 0, 1], R² 0.008 |

Interpretation:

1. **The measured input genuinely helps.** |return| lifts the level R² from
   ~0.01 to 0.04–0.11 and explains up to 12–17% of the long-horizon change
   channels. The input must be the *magnitude*: signed return or return² give
   negative R². R² grows with horizon — the model tracks volume's slow drift,
   not its daily bursts.
2. **The individual aᵢ are not identifiable.** Plain LS needs ±10⁸ cancelling
   values (the three forced responses of |return| are too collinear to
   separate); ridge only tames them by sacrificing R²; imposing aᵢ ≥ 0
   collapses the fit to a pure slow mode with R² ≈ 0. All the predictive
   power lives in the negative-share "difference of smoothings" combinations.
   Only the combined response shape — not the shares — is meaningful.
3. **The dominance pattern from theory does not appear.** The fitted variance
   shares put the 7-day mode at 67–77% in *every* channel (1-day mode only
   2–16%), whereas step-input theory predicts z₁ dominant in Δ₁V. Real
   volume's response to |return| noise is slow-smoothing dominated — mode
   separation requires sharp, step-like excitation (see the event-input
   experiments).
4. **Out-of-sample is mixed**: MSFT's slow component extrapolates (R² +0.14);
   AAPL's does not (R² −1.11).

### Knobs

| Knob | Where |
|---|---|
| Time constants | `TAU` |
| Horizons | `HORIZONS` |
| Input | the `u = ...` line in `build()` |
| Ticker | `run("MSFT")` / `run("AAPL")` — any file with the same format |
| Train/test split | `split = 900` in `run()` |
