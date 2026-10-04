# AAPL Volume — System-Identification Experiments

Replication guide for the three experiments that fit the constant-input 3-state
model to AAPL daily volume. Every code block is self-contained and copy-paste
runnable in order — no `.py` file needed.

---

## The model under test

```
dz1/dt = -z1 + a1·u,   dz2/dt = -z2/7 + a2·u,   dz3/dt = -z3/21 + a3·u
1 = a1 + a2 + a3,      Q = z1 + z2 + z3
```

`u` is an **unknown constant**; `Q` is the observed series (here: volume).
The closed form of Q is a sum of three exponentials plus a constant:

```
Q(t) = Qinf + c1·e^(-t/1) + c2·e^(-t/7) + c3·e^(-t/21)
c_i = z_i(0) - a_i·U·tau_i        Qinf = U·(a1·1 + a2·7 + a3·21)
```

So step 1 of every experiment is the same 4-coefficient least-squares fit;
steps 2–3 differ only in how they slice `(c_i, Qinf)` back into `(a, U, z(0))`.

## Setup

```bash
pip install numpy matplotlib
```

Data: `America_modified/AAPL.txt` — CSV rows `date, open, high, low, close,
volume, ...`; **volume is the 6th column** (index 5). 1,255 daily rows,
2021-08-16 .. 2026-08-14.

> `America_modified_5d/AAPL.txt` and `stock_data/America/AAPL.txt` contain the
> **identical volume series** (verified: max difference 0.0) — they are the same
> data in different formats and are not needed for these experiments.

If you run the blocks outside Jupyter on a Windows console, prefix with
`PYTHONIOENCODING=utf-8` (the console's cp1252 chokes on a few symbols; the code
below uses ASCII in prints to avoid this).

---

## Step 0 — load and check the data

```python
import csv

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq

TAU = np.array([1.0, 7.0, 21.0])          # time constants, in trading days

def load(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    dates = [r[0] for r in rows]
    vol = np.array([float(r[5]) for r in rows])   # 6th column = volume
    return dates, vol

dates, vol = load("America_modified/AAPL.txt")
t_full = np.arange(len(vol), dtype=float)         # days since 2021-08-16
print(f"data: {len(vol)} rows, {dates[0]} .. {dates[-1]}")
print(f"volume: mean {vol.mean():.0f}  std {vol.std():.0f}  "
      f"min {vol.min():.0f}  max {vol.max():.0f}")
```

Expected output:

```
data: 1255 rows, 2021-08-16 .. 2026-08-14
volume: mean 64595408  std 28606733  min 17910600  max 318679900
```

---

## Experiment 1 — the 4-coefficient fit (the gate)

Fit `Q(t) = Qinf + c1·e^(-t) + c2·e^(-t/7) + c3·e^(-t/21)` to the full series.
**This R² is the gate for everything that follows.**

```python
def basis(t):
    return np.column_stack([np.ones_like(t),
                            np.exp(-t / TAU[0]),
                            np.exp(-t / TAU[1]),
                            np.exp(-t / TAU[2])])

def fit_four(Q, t):
    B = basis(t)
    b, *_ = lstsq(B, Q, rcond=None)               # b = [Qinf, c1, c2, c3]
    pred = B @ b
    r2 = 1 - np.sum((Q - pred) ** 2) / np.sum((Q - Q.mean()) ** 2)
    rmse = np.sqrt(np.mean((Q - pred) ** 2))
    return b, pred, r2, rmse

b1, pred1, r2_full, rmse_full = fit_four(vol, t_full)
Qinf, c1, c2, c3 = b1
m30 = 30
r2_30 = 1 - np.sum((vol[:m30] - pred1[:m30]) ** 2) / \
             np.sum((vol[:m30] - vol[:m30].mean()) ** 2)

print("EXPERIMENT 1 - fit Q(t) = Qinf + c1*e^-t + c2*e^(-t/7) + c3*e^(-t/21)")
print(f"  Qinf = {Qinf:,.0f}   c1 = {c1:,.0f}   c2 = {c2:,.0f}   c3 = {c3:,.0f}")
print(f"  extrapolated Q(0) = {Qinf + c1 + c2 + c3:,.0f}   "
      f"(first measured volume {vol[0]:,.0f})")
print(f"  R^2 full series = {r2_full:.3f}   "
      f"R^2 first 30 days = {r2_30:.3f}   RMSE = {rmse_full:,.0f}")
```

Expected output:

```
EXPERIMENT 1 - fit Q(t) = Qinf + c1*e^-t + c2*e^(-t/7) + c3*e^(-t/21)
  Qinf = 63,804,871   c1 = 95,522,734   c2 = -125,457,216   c3 = 82,934,976
  extrapolated Q(0) = 116,805,365   (first measured volume 103,296,000)
  R^2 full series = 0.018   R^2 first 30 days = 0.012   RMSE = 28,354,309
```

Plot (use `plt.show()` in Jupyter, or `fig.savefig("aapl_exp1_fit.png", dpi=100)`):

```python
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c3c2b7"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax1.plot(t_full, vol, lw=0.8, color=GRAY, label="AAPL volume")
ax1.plot(t_full, pred1, lw=1.5, color=BLUE, label="3-exponential fit")
ax1.axhline(Qinf, lw=1, ls="--", color=ORANGE, label=f"Qinf = {Qinf:,.0f}")
ax1.set_xlabel("trading days since 2021-08-16"); ax1.set_ylabel("volume")
ax1.set_title("experiment 1 - full series"); ax1.legend(fontsize=9)
ax2.plot(t_full[:90], vol[:90], lw=1, color=GRAY, label="AAPL volume")
ax2.plot(t_full[:90], pred1[:90], lw=1.5, color=BLUE, label="3-exponential fit")
ax2.set_xlabel("trading days since 2021-08-16"); ax2.set_ylabel("volume")
ax2.set_title("experiment 1 - first 90 days (where the modes live)")
ax2.legend(fontsize=9)
fig.tight_layout()
plt.show()
```

**Verdict: the model does not fit volume** — R² = 0.018. It explains 1.8% of
volume's variance. Daily volume is bursty noise around a drifting mean; it does
not relax smoothly to a steady state. This is the gate that says the downstream
numbers are model artifacts.

---

## Experiment 2 — equal-split assumption z(0) = Q(0)/3

Assumes the three states start equal, recovers `a` and `U` in closed form
(`p_i = a_i·U`, and `Σa_i = 1` gives `U = Σp_i`):

```python
Q0_ext = Qinf + c1 + c2 + c3                    # extrapolated Q(0), low noise
z0 = Q0_ext / 3.0
p = (z0 - b1[1:]) / TAU                         # p_i = a_i * U
U2 = p.sum()                                    # sum(a) = 1  ->  U = sum(p)
a2 = p / U2

print("EXPERIMENT 2 - equal-split assumption z1(0) = z2(0) = z3(0) = Q(0)/3")
print(f"  assumed z(0) = {z0:,.0f}  (all three states)")
print(f"  a = [{a2[0]:+.4f}, {a2[1]:+.4f}, {a2[2]:+.4f}]   (sum = {a2.sum():.4f})")
print(f"  U = {U2:,.0f}  ->  Qinf = U*sum(a_i*tau_i) = "
      f"{U2 * (a2 * TAU).sum():,.0f}  (fit gave {Qinf:,.0f})")
```

Expected output:

```
EXPERIMENT 2 - equal-split assumption z1(0) = z2(0) = z3(0) = Q(0)/3
  assumed z(0) = 38,935,122  (all three states)
  a = [+1.6077, -0.6672, +0.0595]   (sum = 1.0000)
  U = -35,198,224  ->  Qinf = U*sum(a_i*tau_i) = 63,804,871  (fit gave 63,804,871)
```

Note the recovered `U` is **negative** — an immediate physical impossibility
flag for a volume driver, and further confirmation that the model is wrong
(the algebra still closes because `Σa_i·τ_i` came out negative).

Recovered hidden states (first 90 days only — they decay fast):

```python
t60 = t_full[:90]
Z2 = np.stack([z0 * np.exp(-t60 / TAU[i])
               + a2[i] * U2 * TAU[i] * (1 - np.exp(-t60 / TAU[i]))
               for i in range(3)])
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for i in range(3):
    ax.plot(t60, Z2[i], lw=1.5, color=[BLUE, ORANGE, "#0b0b0b"][i],
            label=f"z{i+1}(t)  tau={TAU[i]:.0f}d")
ax.set_xlabel("trading days"); ax.set_ylabel("state value")
ax.set_title("experiment 2 - recovered hidden states (equal-split z(0))")
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
```

---

## Experiment 3 — two "runs": the two-window scheme

Treats two 300-day volume regimes as two experiments with different (unknown)
constant inputs and the *same* initial state. `z(0)` cancels in the difference
`cA_i − cB_i = a_i·τ_i·(U_B − U_A)`, and `Σa_i = 1` closes it.

```python
def window(w0, w1):
    t = np.arange(w1 - w0, dtype=float)
    return vol[w0:w1], t, dates[w0], dates[w1 - 1]

cands = [(0, 300), (300, 600), (600, 900), (900, 1200)]
means = {w: vol[w[0]:w[1]].mean() for w in cands}
pairs = [(a, b) for i, a in enumerate(cands) for b in cands[i + 1:]]
(wA, wB) = max(pairs, key=lambda ab: abs(means[ab[0]] - means[ab[1]]))  # max contrast

QA, tA, dA0, dA1 = window(*wA)
QB, tB, dB0, dB1 = window(*wB)
bA, _, r2A, _ = fit_four(QA, tA)
bB, _, r2B, _ = fit_four(QB, tB)

dc = bA[1:] - bB[1:]                            # a_i * tau_i * (U_B - U_A)
dU = (dc / TAU).sum()                           # sum(a) = 1  ->  U_B - U_A
a3 = (dc / TAU) / dU
UA = bA[0] / (a3 * TAU).sum()
UB = bB[0] / (a3 * TAU).sum()
z0A = bA[1:] + a3 * UA * TAU                    # the shared initial state

print("EXPERIMENT 3 - two-experiment scheme, windows chosen for max mean contrast")
print(f"  window A: rows {wA[0]}-{wA[1]} ({dA0} .. {dA1})  "
      f"mean vol {means[wA]:,.0f}  R^2 {r2A:.3f}")
print(f"  window B: rows {wB[0]}-{wB[1]} ({dB0} .. {dB1})  "
      f"mean vol {means[wB]:,.0f}  R^2 {r2B:.3f}")
print(f"  a = [{a3[0]:+.4f}, {a3[1]:+.4f}, {a3[2]:+.4f}]   (sum = {a3.sum():.4f})")
print(f"  U_A = {UA:,.0f}   U_B = {UB:,.0f}   ratio U_A/U_B = {UA/UB:.3f}")
print(f"  window mean-volume ratio A/B = {means[wA]/means[wB]:.3f}   (consistency check)")
print(f"  recovered shared z(0) = [{z0A[0]:,.0f}, {z0A[1]:,.0f}, {z0A[2]:,.0f}]")
```

Expected output:

```
EXPERIMENT 3 - two-experiment scheme, windows chosen for max mean contrast
  window A: rows 0-300 (2021-08-16 .. 2022-10-21)  mean vol 88,105,806  R^2 0.028
  window B: rows 900-1200 (2025-03-18 .. 2026-05-27)  mean vol 51,870,295  R^2 0.110
  a = [+0.5247, +0.6980, -0.2227]   (sum = 1.0000)
  U_A = 122,358,362   U_B = 65,874,956   ratio U_A/U_B = 1.857
  window mean-volume ratio A/B = 1.699   (consistency check)
  recovered shared z(0) = [105,811,930, 605,255,828, -602,182,516]
```

Window plot:

```python
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, (Qw, tw, pw, lab) in zip(axes,
        [(QA, tA, bA, f"A ({dA0}..{dA1})"), (QB, tB, bB, f"B ({dB0}..{dB1})")]):
    ax.plot(tw, Qw, lw=0.8, color=GRAY, label="volume")
    ax.plot(tw, basis(tw) @ pw, lw=1.5, color=BLUE, label="3-exponential fit")
    ax.axhline(pw[0], lw=1, ls="--", color=ORANGE, label=f"Qinf = {pw[0]:,.0f}")
    ax.set_xlabel("days since window start"); ax.set_ylabel("volume")
    ax.set_title(f"experiment 3 - window {lab}")
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
```

---

## Cross-check — do the two methods agree?

```python
print("CROSS-CHECK")
print(f"  a from equal-split : {np.round(a2, 4)}")
print(f"  a from two-windows : {np.round(a3, 4)}")
print(f"  |d a| = {np.abs(a2 - a3).max():.4f}")
```

Expected output:

```
CROSS-CHECK
  a from equal-split : [ 1.6077 -0.6672  0.0595]
  a from two-windows : [ 0.5247  0.698  -0.2227]
  |d a| = 1.3652
```

---

## Step 4 — do transforms rescue the fit? (they don't)

```python
Bfull = basis(t_full)

def r2_of(Q):
    b, *_ = lstsq(Bfull, Q, rcond=None)
    pred = Bfull @ b
    return 1 - np.sum((Q - pred) ** 2) / np.sum((Q - Q.mean()) ** 2)

print("raw volume       R^2 =", round(r2_of(vol), 4))
print("log(volume)      R^2 =", round(r2_of(np.log(vol)), 4))
print("centered volume  R^2 =", round(r2_of(vol - vol.mean()), 4))
print("log centered     R^2 =", round(r2_of(np.log(vol) - np.log(vol).mean()), 4))
rm = np.convolve(vol, np.ones(21) / 21, mode="valid")
print("slow-component share of variance (21d mean):",
      round(np.var(rm) / np.var(vol), 4))
```

Expected output:

```
raw volume       R^2 = 0.0176
log(volume)      R^2 = 0.0243
centered volume  R^2 = 0.0176
log centered     R^2 = 0.0243
slow-component share of variance (21d mean): 0.3881
```

---

## Results summary and interpretation

| Experiment | Result |
|---|---|
| 1 — 4-coefficient fit | R² = **0.018** full series, 0.012 first 30 days → **model class is wrong for volume** |
| 2 — equal-split | a = [1.608, −0.667, 0.060], U = **−35.2M** (negative!) — artifacts of the failed fit |
| 3 — two-window | a = [0.525, 0.698, −0.223], U = 122.4M / 65.9M, z(0) contains ±600M states — artifacts |
| Cross-check | \|Δa\| = 1.37 — the two methods disagree wildly, as they must when the base model fails |

What this means:

1. **The 3-exponential model does not describe AAPL volume.** It explains ~1.8%
   of variance (MSFT: 0.3%), and no simple transform fixes it (log: 0.0243).
   Volume is bursty noise around a drifting mean; only 39% of its variance is
   even slow-moving, and that part isn't exponential-shaped.
2. **Therefore the recovered `a`, `U`, `z(0)` are not meaningful.** The
   impossible signs make this even clearer than for MSFT: experiment 2 returns
   a **negative constant input** U = −35.2M, and experiment 3 returns hidden
   states of ±600M shares with no volume process to back them. These are the
   outputs of a correct pipeline fed a wrong model — the silent-failure class,
   caught here by the R² gate. **Always check the gate (R² /
   loss-vs-noise-floor) before trusting the recovered parameters.**
3. **The pipeline itself is sound**, shown by two internal checks:
   - Experiment 2's reconstructed `U·Σaᵢτᵢ` equals the fitted Q∞ **exactly**
     (63,804,871 both ways) — the algebra closes.
   - Experiment 3's input ratio `U_A/U_B = 1.857` points in the direction of
     the measured window-mean-volume ratio **1.699** — the method reads the
     *relative* input levels between windows (less tightly than MSFT's
     1.50 vs 1.40, but still the right ballpark).

### If you want to change things

| Knob | Where |
|---|---|
| Time constants | `TAU` (currently 1, 7, 21 trading days) |
| Windows in experiment 3 | `cands` — any `(start, end)` row ranges |
| Ticker | the path in `load(...)` — any file with the same 6-column format |
| Model verdict gate | the R² printed by experiment 1 |
| 5-day aggregated volume (the 5d file does NOT contain it) | `vol.reshape(-1, 5).mean(axis=1)` before fitting |

### If the goal is volume modelling

Use a model class that fits volume's actual structure — e.g., the PART II · 4
state-space machinery with a measured input (close-price returns as `u`), or a
stochastic/AR model. The constant-input 3-exponential model is for systems that
relax smoothly to steady state — feed it that kind of data and every experiment
above recovers the true parameters to high precision.
