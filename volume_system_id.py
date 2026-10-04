# ╔══════════════════════════════════════════════════════════════════╗
# ║   MSFT volume — system-identification experiments                ║
# ╚══════════════════════════════════════════════════════════════════╝
# Model under test (u = UNKNOWN CONSTANT):
#     dz1/dt = -z1 + a1·u,  dz2/dt = -z2/7 + a2·u,  dz3/dt = -z3/21 + a3·u
#     1 = a1 + a2 + a3,     Q = z1 + z2 + z3
# Closed form of Q:  Q(t) = Qinf + c1·e^(-t/1) + c2·e^(-t/7) + c3·e^(-t/21)
# with c_i = z_i(0) - a_i·U·tau_i  and  Qinf = U·(a1·1 + a2·7 + a3·21).
#
# Experiments (Q = MSFT daily volume, 6th column; t in trading days):
#   1. Fit the 4-coefficient model to the full series -> c_i, Qinf, R².
#   2. Equal-split assumption z(0) = Q(0)/3 -> recover a1..a3 and U.
#   3. Two-experiment scheme: two 300-day windows as two "runs" with
#      different (unknown) constant inputs -> recover a, U1, U2, z(0).
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq

# ---- palette consistent with the notebook -----------------------------------
BLUE, ORANGE, BLACK, GRAY = "#2a78d6", "#eb6834", "#0b0b0b", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "axes.titlecolor": "#0b0b0b", "xtick.color": "#898781",
    "ytick.color": "#898781", "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
})

TAU = np.array([1.0, 7.0, 21.0])          # time constants, in days

# ---- load: volume is the 6th column (index 5) --------------------------------
def load(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    dates = [r[0] for r in rows]
    vol = np.array([float(r[5]) for r in rows])
    return dates, vol

dates, vol = load("America_modified/MSFT.txt")
t_full = np.arange(len(vol), dtype=float)         # days since 2021-08-16
print(f"data: {len(vol)} rows, {dates[0]} .. {dates[-1]}")
print(f"volume: mean {vol.mean():.0f}  std {vol.std():.0f}  min {vol.min():.0f}  max {vol.max():.0f}")

def basis(t):
    return np.column_stack([np.ones_like(t),
                            np.exp(-t / TAU[0]), np.exp(-t / TAU[1]), np.exp(-t / TAU[2])])

def fit_four(Q, t):
    B = basis(t)
    b, *_ = lstsq(B, Q, rcond=None)               # b = [Qinf, c1, c2, c3]
    pred = B @ b
    r2 = 1 - np.sum((Q - pred) ** 2) / np.sum((Q - Q.mean()) ** 2)
    rmse = np.sqrt(np.mean((Q - pred) ** 2))
    return b, pred, r2, rmse

# ═══════════ EXPERIMENT 1 — the 4-coefficient fit on the full series ═══════════
b1, pred1, r2_full, rmse_full = fit_four(vol, t_full)
Qinf, c1, c2, c3 = b1
m30 = 30
r2_30 = 1 - np.sum((vol[:m30] - pred1[:m30]) ** 2) / np.sum((vol[:m30] - vol[:m30].mean()) ** 2)
print("\nEXPERIMENT 1 — fit Q(t) = Qinf + c1·e^-t + c2·e^(-t/7) + c3·e^(-t/21)")
print(f"  Qinf = {Qinf:,.0f}   c1 = {c1:,.0f}   c2 = {c2:,.0f}   c3 = {c3:,.0f}")
print(f"  extrapolated Q(0) = {Qinf + c1 + c2 + c3:,.0f}   (first measured volume {vol[0]:,.0f})")
print(f"  R² full series = {r2_full:.3f}   R² first 30 days = {r2_30:.3f}   RMSE = {rmse_full:,.0f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax1.plot(t_full, vol, lw=0.8, color=GRAY, label="MSFT volume")
ax1.plot(t_full, pred1, lw=1.5, color=BLUE, label="3-exponential fit")
ax1.axhline(Qinf, lw=1, ls="--", color=ORANGE, label=f"Q∞ = {Qinf:,.0f}")
ax1.set_xlabel("trading days since 2021-08-16"); ax1.set_ylabel("volume")
ax1.set_title("experiment 1 — full series")
ax1.legend(fontsize=9)
ax2.plot(t_full[:90], vol[:90], lw=1, color=GRAY, label="MSFT volume")
ax2.plot(t_full[:90], pred1[:90], lw=1.5, color=BLUE, label="3-exponential fit")
ax2.set_xlabel("trading days since 2021-08-16"); ax2.set_ylabel("volume")
ax2.set_title("experiment 1 — first 90 days (where the modes live)")
ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig("msft_exp1_fit.png", dpi=100)

# ═══════════ EXPERIMENT 2 — equal-split z(0) = Q(0)/3 → recover a, U ═══════════
Q0_ext = Qinf + c1 + c2 + c3                     # extrapolated Q(0), low noise
z0 = Q0_ext / 3.0
p = (z0 - b1[1:]) / TAU                          # p_i = a_i·U
U2 = p.sum()                                     # Σ a_i = 1  →  U = Σ p_i
a2 = p / U2
print("\nEXPERIMENT 2 — equal-split assumption z1(0) = z2(0) = z3(0) = Q(0)/3")
print(f"  assumed z(0) = {z0:,.0f}  (all three states)")
print(f"  a = [{a2[0]:+.4f}, {a2[1]:+.4f}, {a2[2]:+.4f}]   (sum = {a2.sum():.4f})")
print(f"  U = {U2:,.0f}  →  Q∞ = U·Σa_i·τ_i = {U2 * (a2 * TAU).sum():,.0f}  (fit gave {Qinf:,.0f})")

t60 = t_full[:90]
Z2 = np.stack([z0 * np.exp(-t60 / TAU[i]) + a2[i] * U2 * TAU[i] * (1 - np.exp(-t60 / TAU[i]))
               for i in range(3)])
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for i in range(3):
    ax.plot(t60, Z2[i], lw=1.5, color=[BLUE, ORANGE, BLACK][i], label=f"z{i+1}(t)  τ={TAU[i]:.0f}d")
ax.axhline(U2 * a2 @ TAU, lw=1, ls=":", color=GRAY, label="steady states aᵢ·U·τᵢ")
ax.set_xlabel("trading days"); ax.set_ylabel("state value")
ax.set_title("experiment 2 — recovered hidden states (equal-split z(0))")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("msft_exp2_states.png", dpi=100)

# ═══════════ EXPERIMENT 3 — two "runs": two 300-day windows ═══════════
# (Treats two different volume regimes as two experiments with different
#  constant inputs U and the SAME initial state — the model's assumption.)
def window(w0, w1):
    t = np.arange(w1 - w0, dtype=float)
    return vol[w0:w1], t, dates[w0], dates[w1 - 1]

cands = [(0, 300), (300, 600), (600, 900), (900, 1200)]
means = {w: vol[w[0]:w[1]].mean() for w in cands}
pairs = [(a, b) for i, a in enumerate(cands) for b in cands[i + 1:]]
(wA, wB) = max(pairs, key=lambda ab: abs(means[ab[0]] - means[ab[1]]))
QA, tA, dA0, dA1 = window(*wA)
QB, tB, dB0, dB1 = window(*wB)
bA, _, r2A, _ = fit_four(QA, tA)
bB, _, r2B, _ = fit_four(QB, tB)
dc = bA[1:] - bB[1:]                             # a_i·τ_i·(U_B − U_A), z(0) cancels
dU = (dc / TAU).sum()
a3 = (dc / TAU) / dU
UA = bA[0] / (a3 * TAU).sum()
UB = bB[0] / (a3 * TAU).sum()
z0A = bA[1:] + a3 * UA * TAU                     # the shared initial state
print(f"\nEXPERIMENT 3 — two-experiment scheme, windows chosen for max mean contrast")
print(f"  window A: rows {wA[0]}-{wA[1]} ({dA0} .. {dA1})  mean vol {means[wA]:,.0f}  R² {r2A:.3f}")
print(f"  window B: rows {wB[0]}-{wB[1]} ({dB0} .. {dB1})  mean vol {means[wB]:,.0f}  R² {r2B:.3f}")
print(f"  a = [{a3[0]:+.4f}, {a3[1]:+.4f}, {a3[2]:+.4f}]   (sum = {a3.sum():.4f})")
print(f"  U_A = {UA:,.0f}   U_B = {UB:,.0f}   ratio U_A/U_B = {UA/UB:.3f}")
print(f"  window mean-volume ratio A/B = {means[wA]/means[wB]:.3f}   (consistency check)")
print(f"  recovered shared z(0) = [{z0A[0]:,.0f}, {z0A[1]:,.0f}, {z0A[2]:,.0f}]")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, (Qw, tw, pw, lab) in zip(axes,
        [(QA, tA, bA, f"A ({dA0}..{dA1})"), (QB, tB, bB, f"B ({dB0}..{dB1})")]):
    ax.plot(tw, Qw, lw=0.8, color=GRAY, label="volume")
    ax.plot(tw, basis(tw) @ pw, lw=1.5, color=BLUE, label="3-exponential fit")
    ax.axhline(pw[0], lw=1, ls="--", color=ORANGE, label=f"Q∞ = {pw[0]:,.0f}")
    ax.set_xlabel("days since window start"); ax.set_ylabel("volume")
    ax.set_title(f"experiment 3 — window {lab}")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("msft_exp3_windows.png", dpi=100)

# ═══════════ cross-check: do the two methods agree? ═══════════
print("\nCROSS-CHECK")
print(f"  a from equal-split : {np.round(a2, 4)}")
print(f"  a from two-windows : {np.round(a3, 4)}")
print(f"  |Δa| = {np.abs(a2 - a3).max():.4f}")
