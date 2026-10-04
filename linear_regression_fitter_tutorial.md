# Tutorial: Joint Linear Regression Fitter & PyTorch ODE Solver

This document is the markdown version of `linear_regression_fitter.ipynb` — every cell
is reproduced below, with a tutorial on **setup**, **where to change the equations and
input data**, and **how long training takes**.

The notebook has two independent halves:

| | What it does | Fitting method | Runtime |
|---|---|---|---|
| **PART I** | Fit 6 shared parameters across 3 linear equations using 3 data sources | `sklearn` `LinearRegression` (closed form) | < 1 s |
| **PART II** | Solve ODEs (scalar → N-dim systems) and learn ODE parameters from noisy data by backpropagating **through the solver** | Adam (gradient descent) | seconds to a few minutes |

---

## 1. Setup

### 1.1 Packages

Python 3.9+ with:

```
pip install numpy pandas scikit-learn matplotlib torch
```

* **PART I** needs `numpy pandas scikit-learn matplotlib`.
* **PART II** needs `torch matplotlib`.
* This repository already ships a `.venv` with everything installed
  (torch 2.13.0+cpu). To reuse it:

```
.venv\Scripts\activate        # Windows
```

### 1.2 Running it

* **Option A — Jupyter (recommended):** open the notebook and *Run All*.
  Cells build on each other (PART II · 4 reuses functions defined in
  PART II · 1–2), so keep them in order.
* **Option B — VS Code:** open the `.ipynb` with the Python/Jupyter extension, Run All.
* **Option C — plain script:** copy the code blocks below, in order, into a `.py` file
  and run it. Replace `plt.show()` with `plt.savefig(...)` if you have no display.

> CPU-only torch is fine — these problems are small; a GPU would not change the
> runtimes below meaningfully.

---

## 2. PART I — Joint linear regression: 3 equations, 6 unknowns

Three equations share 6 unknown parameters (`a`…`f`). The fitter stacks the data
from all 3 sources and fits all 6 parameters in **one** joint least-squares fit.

### ✏️ 2.1 Where you change the equations

Edit the `EQUATIONS` list — one tuple per equation: `(equation, target column, data source)`.

```python
EXPECTED_UNKNOWNS = 6   # how many unknown parameters the equations share

EQUATIONS = [
    # (equation,                target, data source)
    ("a + b*x + c*x**2",        "y", "source1"),
    ("a + d*t + e*t**2",        "y", "source2"),
    ("c*w + f*sin(w) + e*w",    "y", "source3"),
]
```

Rules:

* **PARAMETERS** = the single letters you want fitted (`a`…`f`). They are **shared**
  across the three equations — that is the whole point of the joint fit.
* **VARIABLES** = any other name (`x`, `t`, `w`) — it must match a column in that
  equation's data source.
* **TARGET** = the column you are predicting (`"y"` above).
* Each equation must be **linear in its parameters**:
  * OK: `"a + b*x"`, `"a + b*log(x)"`, `"a*sin(x) + b*cos(x)"`
  * NOT OK: `"a*exp(b*x)"` (b is inside `exp`) → use `scipy.optimize.curve_fit` instead.
* Don't use `e` or `pi` as parameter names (reserved for the constants).
* Set `EXPECTED_UNKNOWNS` to the number of distinct parameters your equations use.

### ✏️ 2.2 Where you change the input data

Edit the `DATA_SOURCES` dict — one entry per source named in `EQUATIONS`:

```python
DATA_SOURCES = {
    "source1": {  # y = 1.5 + 2·x + 3·x² + noise
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [6.6, 17.3, 34.7, 57.2, 86.9, 121.3, 162.8, 209.2, 262.9, 321.1],
    },
    "source2": {  # y = 1.5 + 4·t + 0.5·t² + noise
        "t": [1, 2, 3, 4, 5, 6, 7, 8],
        "y": [6.1, 11.4, 18.2, 25.4, 34.2, 43.7, 53.8, 65.7],
    },
    "source3": {  # y = 3.5·w + 6·sin(w) + noise
        "w": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        "y": [4.7, 8.6, 11.3, 12.4, 12.4, 11.3, 10.2, 9.5, 9.8, 11.8, 15.0, 19.4],
    },
}
```

* Sources may have **different lengths** — they are stacked internally.
* Two equations may share a source, and any equation may use any source.
* To load real data from CSVs instead:

```python
import pandas as pd
df1 = pd.read_csv("ticker_A.csv")
DATA_SOURCES = {
    "source1": {col: df1[col].to_numpy() for col in df1.columns},
    # ...same for source2, source3
}
```

### 2.3 The model (no edits needed)

Parses the equations, detects the shared parameters, builds one stacked design
matrix, runs one `LinearRegression(fit_intercept=False)` fit, prints the fitted
values plus per-equation R²/RMSE, then runs a sanity check that every equation is
really linear in its parameters:

```python
import re
import numpy as np
from sklearn.linear_model import LinearRegression

ns = {   # functions you may use inside the equations
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "log2": np.log2, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "pow": np.power,
    "pi": np.pi, "e": np.e,
}

# ---- 1. Parse the three equations -----------------------------------------
parsed = []
all_ids = set()
targets = set()
for i, (eq, target, source) in enumerate(EQUATIONS, start=1):
    if source not in DATA_SOURCES:
        raise ValueError(f"Equation {i} uses '{source}' but DATA_SOURCES has no such key.")
    if target not in DATA_SOURCES[source]:
        raise ValueError(f"Equation {i}: column '{target}' missing from '{source}'.")
    eq = eq.strip()
    if "=" in eq:
        eq = eq.split("=", 1)[1].strip()
    all_ids |= set(re.findall(r"[A-Za-z_]\w*", eq))
    targets.add(target)
    parsed.append((eq, target, source))

var_names = {name
             for _, _, source in parsed
             for name in DATA_SOURCES[source] if name in all_ids}
params = sorted(all_ids - set(ns) - var_names - targets)
print(f"Parameters     : {params}  ({len(params)} detected, expected {EXPECTED_UNKNOWNS})")
if len(params) != EXPECTED_UNKNOWNS:
    print(f"⚠ Expected {EXPECTED_UNKNOWNS} unknowns but found {len(params)} — check your equations.")

# ---- 2. Build the stacked design matrix (one column per parameter) --------
def _term_for(eq_text, source, param):
    local = dict(ns)
    for name in DATA_SOURCES[source]:
        local[name] = np.asarray(DATA_SOURCES[source][name], dtype=float)
    for p in params:
        local[p] = 1.0 if p == param else 0.0
    return np.asarray(eval(eq_text, {"__builtins__": {}}, local), dtype=float)

X_blocks, y_blocks = [], []
for eq_text, target, source in parsed:
    X_blocks.append(np.column_stack([_term_for(eq_text, source, p) for p in params]))
    y_blocks.append(np.asarray(DATA_SOURCES[source][target], dtype=float))

X = np.vstack(X_blocks)
y = np.concatenate(y_blocks)
print(f"Total samples  : {len(y)}")

# ---- 3. One joint fit across all equations and sources ---------------------
model = LinearRegression(fit_intercept=False)  # the 6 unknowns are the columns of X
model.fit(X, y)

# ---- 4. Fitted unknowns + per-equation metrics -----------------------------
def _r2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

def _rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

print("\nFitted unknowns:")
for p, c in zip(params, model.coef_):
    print(f"  {p} = {c:.4g}")

print("\nPer-equation fit (using the shared parameters):")
eq_summaries = []
for k, ((eq_text, target, source), Xb, yb) in enumerate(zip(parsed, X_blocks, y_blocks), start=1):
    pred = model.predict(Xb)
    eq_summaries.append((k, source, target, yb, pred))
    print(f"  Eq {k} [{source}] : R² = {_r2(yb, pred):.4f}   RMSE = {_rmse(yb, pred):.4f}")

print(f"\nOverall R² = {_r2(y, model.predict(X)):.4f}")

# ---- 5. Sanity check: are the equations really linear in the parameters? ---
ok = True
for k, (eq_text, target, source) in enumerate(parsed, start=1):
    local = dict(ns)
    for name in DATA_SOURCES[source]:
        local[name] = np.asarray(DATA_SOURCES[source][name], dtype=float)
    for p, c in zip(params, model.coef_):
        local[p] = c
    recon = np.asarray(eval(eq_text, {"__builtins__": {}}, local), dtype=float)
    Xb = np.column_stack([_term_for(eq_text, source, p) for p in params])
    if not np.allclose(recon, Xb @ model.coef_, atol=1e-6):
        ok = False
        print(f"⚠ Equation {k} ('{eq_text}') looks NON-linear in its parameters.")
if ok:
    print("\n✓ Linear-in-parameters check passed for all equations — the fit is valid.")
```

Followed by one diagnostic plot per equation (actual vs predicted):

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "axes.titlecolor": "#0b0b0b", "xtick.color": "#898781",
    "ytick.color": "#898781", "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, axes = plt.subplots(1, len(eq_summaries), figsize=(5 * len(eq_summaries), 4.2))

for ax, (k, source, target, yb, pred) in zip(axes, eq_summaries):
    ax.scatter(yb, pred, s=24, alpha=0.8, color="#2a78d6",
               edgecolors="white", linewidths=0.5)
    lo, hi = min(yb.min(), pred.min()), max(yb.max(), pred.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            ls="--", lw=1.2, color="#c3c2b7")
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"Equation {k} — {source}")

fig.tight_layout()
plt.show()
```

### ⏱ 2.4 Expected runtime

**Under a second.** One sklearn fit on ~30 rows — there is no "training" here.

---

## 3. PART II — Solving ODEs with PyTorch (scalar + system + fitting)

Every integrator step is plain torch math, so the whole simulated trajectory is one
differentiable computation graph. Two payoffs:

* solve any `dz/dt = f(t, z)` with autograd tracking (Euler or RK4);
* backprop **through** the solver → learn unknown ODE parameters from data.
  The ODE is *not* linear in its parameters, so this is where gradient descent
  replaces PART I's `LinearRegression`.

### 3.1 Cell 1 — the ODE solver machinery (scalar)

Defines `solve_ode` (used by every later cell) and checks the scalar case
`dz/dt = p·z + a·u` against its analytic solution:

```python
import matplotlib.pyplot as plt
import torch

plt.rcParams.update({   # same style as the regression plots in PART I
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "axes.titlecolor": "#0b0b0b", "xtick.color": "#898781",
    "ytick.color": "#898781", "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---- the machinery: a differentiable Euler / RK4 integrator -----------------
def solve_ode(f, z0, t, *params, method="rk4"):
    """Integrate dz/dt = f(t, z, *params) on a uniform grid t.

    Every step is plain torch math, so autograd flows THROUGH the solver —
    that is what lets the later cells learn ODE parameters straight from data.
    """
    dt = t[1] - t[0]
    z, zs = z0, [z0]
    for ti in t[:-1]:
        if method == "euler":
            z = z + dt * f(ti, z, *params)
        elif method == "rk4":
            k1 = f(ti,          z,              *params)
            k2 = f(ti + dt / 2, z + dt / 2 * k1, *params)
            k3 = f(ti + dt / 2, z + dt / 2 * k2, *params)
            k4 = f(ti + dt,     z + dt * k3,     *params)
            z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown method {method!r}")
        zs.append(z)
    return torch.stack(zs)

# ---- the ODE and its analytic solution ---------------------------------------
def rhs_scalar(t, z, p, a, u):      # dz/dt = p·z + a·u   (constant input u)
    return p * z + a * u

p, a, u = -0.5, 1.0, 2.0            # p < 0 → stable; steady state z* = −a·u/p = 4
z0 = torch.tensor(1.0)

t = torch.linspace(0, 10, 101)      # coarse grid, dt = 0.1

z_euler = solve_ode(rhs_scalar, z0, t, p, a, u, method="euler")
z_rk4   = solve_ode(rhs_scalar, z0, t, p, a, u, method="rk4")
z_exact = (z0 + a * u / p) * torch.exp(p * t) - a * u / p     # analytic solution

print(f"final value   Euler = {z_euler[-1].item():.6f} | RK4 = {z_rk4[-1].item():.6f} | exact = {z_exact[-1].item():.6f}")
print(f"max |error|   Euler = {(z_euler - z_exact).abs().max().item():.3e} | RK4 = {(z_rk4 - z_exact).abs().max().item():.3e}")
# ... plus trajectory and log-error plots
```

✏️ Change `p, a, u`, `z0`, or the grid `t` to play with the scalar ODE. No edits
are required for the cells that follow.

### 3.2 Cell 2 — the GENERALISED system: `dz/dt = A·z + B·u(t)`

N coupled differential equations with an M-dim time-varying input, written as one
matrix equation:

```
dz1/dt = A11·z1 + A12·z2 + … + A1N·zN + B1·u(t)
dz2/dt = A21·z1 + A22·z2 + … + A2N·zN + B2·u(t)
  ⋮
dzN/dt = AN1·z1 + … + ANN·zN + BN·u(t)
```

```python
import torch

N_STATES, N_INPUTS = 3, 2          # ⬅ the generalisation lives in these two numbers

def make_stable_matrix(n, margin=0.3):
    """Random n×n matrix with Re(λ) ≤ −margin (asymptotically stable)."""
    M = torch.randn(n, n)
    shift = torch.linalg.eigvals(M).real.max() + margin
    return M - shift * torch.eye(n)

def input_signal(t):
    """M-dim time-varying input: one sinusoid per channel, distinct freqs."""
    k = torch.arange(1, N_INPUTS + 1)
    return torch.sin(0.8 * k * t)

def rhs_system(t, z, A, B):        # dz/dt = A·z + B·u(t)
    return A @ z + B @ input_signal(t)

torch.manual_seed(0)               # reproducible random stable system
A = make_stable_matrix(N_STATES)
B = torch.randn(N_STATES, N_INPUTS) * 0.5
z0 = torch.randn(N_STATES) * 0.5

print(f"system: {N_STATES} state equations × {N_INPUTS} inputs,  "
      f"max Re(λ(A)) = {torch.linalg.eigvals(A).real.max().item():+.3f}  (stable if < 0)")

t     = torch.linspace(0, 12, 121)   # dt = 0.1
t_fin = torch.linspace(0, 12, 1201)  # 10× finer reference grid

z_euler = solve_ode(rhs_system, z0, t,     A, B, method="euler")
z_rk4   = solve_ode(rhs_system, z0, t,     A, B, method="rk4")
z_ref   = solve_ode(rhs_system, z0, t_fin, A, B, method="rk4")[::10]

print(f"max |error| vs fine RK4   Euler = {(z_euler - z_ref).abs().max().item():.3e}   RK4 = {(z_rk4 - z_ref).abs().max().item():.3e}")
# ... plus state-trajectory plot and a phase portrait
# (2-D drift-field quiver for N = 2, a 3-D trajectory curve for N ≥ 3)
```

**✏️ Where to change things:**

* `N_STATES`, `N_INPUTS` — any values work; the code is written generically.
* Want your **own** system instead of a random one? Replace the block that builds
  `A`, `B`, `z0` with your numbers, e.g.:

  ```python
  A = torch.tensor([[-0.3, 0.5, 0.1],
                    [-0.5, -0.7, 0.2],
                    [0.1, -0.2, -0.4]])
  B = torch.tensor([[1.0], [0.5], [0.2]])   # N_STATES × N_INPUTS
  z0 = torch.tensor([1.0, -0.5, 0.3])
  ```
  (If you hand-write `A`, check its eigenvalues are stable — `Re(λ) < 0` — or the
  trajectories will explode.)
* Change the input in `input_signal(t)` — e.g. replace with a step, ramp, or your
  own data-driven `u(t)`.

**Reading the output:** the two error numbers quantify solver accuracy
(Euler ≈ 0.09 with dt=0.1, RK4 ≈ 5e-6 — RK4 is effectively exact here). There is
no closed-form solution for a time-varying input, which is why the reference is a
10× finer RK4 run.

### 3.3 Cell 3 — learn `p` and `a` from data (gradient descent through RK4)

The scalar ODE is **not** linear in `p`, `a` — but because `solve_ode` is a chain of
differentiable torch ops, Adam can fit them like network weights:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# ---- 1. TARGET: the measured trajectory the ODE is fitted to --------------------
# (here: synthetic truth + noise; replace z_data with your own measurements)
p_true, a_true = -0.5, 1.0
t_obs = torch.linspace(0, 5, 26)
z_data = solve_ode(rhs_scalar, torch.tensor(1.0), t_obs, p_true, a_true, u) \
       + 0.01 * torch.randn(26)

# ---- 2. learn p, a with Adam --------------------------------------------------
params = nn.Parameter(torch.tensor([-0.1, 0.2]))    # deliberately bad guess
opt = torch.optim.Adam([params], lr=0.05)

for it in range(400):
    p, a = params[0], params[1]
    z_pred = solve_ode(rhs_scalar, torch.tensor(1.0), t_obs, p, a, u)
    loss = ((z_pred - z_data) ** 2).mean()
    opt.zero_grad()
    loss.backward()          # gradients flow back through 25 RK4 steps
    opt.step()
    if it % 100 == 0:
        print(f"  iter {it:3d}   p = {p.item():+.4f}   a = {a.item():+.4f}   loss = {loss.item():.3e}")

print(f"\n  true p = {p_true:+.1f}, a = {a_true:+.1f}"
      f"   |   recovered p = {params[0].item():+.4f}, a = {params[1].item():+.4f}")
# ... plus fitted-vs-measured plot
```

**✏️ Where to change things:** `p_true, a_true`, the noise level (`0.01 *`),
`t_obs`, and the number of iterations. Starting guess `[-0.1, 0.2]` is deliberately
bad — Adam recovers `p ≈ −0.5`, `a ≈ 1.0` in ~400 iterations.

### 3.4 Cell 4 — the GENERALISED mixed system: N ODEs + N linear equations

A full hidden state-space model. **N hidden states** follow N coupled differential
equations; **N linear equations** map them to the measurements:

```
dz/dt = A·z + B·u(t)          (N differential equations → solve_ode)
y1 = c1·z1 + d  …  yN = cN·zN + d    (N linear equations)
```

The system is run **twice from two known initial states** (E = 2 experiments).
Why two? A single trajectory identifies the model only *up to a similarity
transform* `T` with `T·z0 = z0` — in N ≥ 2 dimensions that ambiguity is real, and
two different `z0`s pin it down. The gains `c` and the shared offset `d` are
unknown, exactly like PART I's shared parameters.

```python
import torch
import torch.nn as nn

torch.manual_seed(1)                        # reproducible truth + data

N = 3                                       # ⬅ dimension knobs: states...
E = 2                                       # ...and experiments
NOISE = 0.01
t_obs = torch.linspace(0, 10, 51)           # where the outputs are observed

# ---- 1. the true system -------------------------------------------------------
A_t = make_stable_matrix(N, margin=0.1)     # general N×N coupling (hidden)
B_t = torch.randn(N, N_INPUTS) * 0.5
c_t = torch.randn(N) * 0.8
d_t = 0.5 * torch.randn(1).item()
Z0 = torch.randn(E, N) * 0.5                # E KNOWN initial states
Z0[1] += 0.3                                #   (kept distinct)

def mixed_forward(t, A, B, c, d):
    zs, ys = [], []
    for z0e in Z0:
        z = solve_ode(rhs_system, z0e, t, A, B)     # N hidden states
        zs.append(z)
        ys.append(z * c + d)                        # N linear output equations
    return torch.stack(zs), torch.stack(ys)         # (E, T, N) each

t_sim = torch.linspace(0, 10, 201)
Z_true, Y_true = mixed_forward(t_sim, A_t, B_t, c_t, d_t)

# ---- 2. TARGET: the observed outputs, shape (E, len(t_obs), N) -----------------
# (here: synthetic truth + noise. For real data, replace Y_obs with your
#  measurements and set Z0 to the initial states the experiments actually
#  started from — then comment out the truth comparison in section 5.)
torch.manual_seed(101)
Y_obs = mixed_forward(t_obs, A_t, B_t, c_t, d_t)[1] + NOISE * torch.randn(E, 51, N)
print(f"noise floor (σ²) = {NOISE**2:.1e} — the target loss")

# ---- 3. the linear part: exact least squares for c, d (closed form) ------------
def fit_cd(Z, Y):
    """Least squares for y_i(t) = c_i·z_i(t) + d — differentiable, exact."""
    T, _ = Z.shape
    rows = []
    for i in range(N):
        R = torch.zeros(T, N + 1)
        R[:, i] = Z[:, i]
        R[:, -1] = 1.0
        rows.append(R)
    sol = torch.linalg.lstsq(torch.cat(rows), Y.T.reshape(-1)).solution
    return sol[:N], sol[-1]

# ---- 4. learn A, B with Adam + restarts (keep the best loss) -------------------
def loss_fn(par):
    A = par[:N*N].reshape(N, N)
    B = par[N*N:].reshape(N, N_INPUTS)
    Z = torch.cat([solve_ode(rhs_system, z0e, t_obs, A, B) for z0e in Z0])
    Y = Y_obs.reshape(-1, N)
    c, d = fit_cd(Z, Y)
    return ((Z * c + d - Y) ** 2).mean()

best_par, best_loss = None, float("inf")
torch.manual_seed(201)
for start in range(8):                     # local minima → keep the best run
    par = nn.Parameter(torch.cat([
        make_stable_matrix(N).flatten() * 0.3,
        torch.randn(N, N_INPUTS).flatten() * 0.3,
    ]))
    opt = torch.optim.Adam([par], lr=0.05)
    for it in range(500):
        loss = loss_fn(par)
        opt.zero_grad(); loss.backward(); opt.step()
    if loss.item() < best_loss:
        best_loss, best_par = loss.item(), par.detach().clone()
    print(f"  start {start}: final loss = {loss.item():.3e}"
          + ("  ← best" if loss.item() <= best_loss else ""))

A_f = best_par[:N*N].reshape(N, N)
B_f = best_par[N*N:].reshape(N, N_INPUTS)
Z_fit_obs = torch.cat([solve_ode(rhs_system, z0e, t_obs, A_f, B_f) for z0e in Z0])
c_f, d_f = fit_cd(Z_fit_obs, Y_obs.reshape(-1, N))

# ---- 5. report -------------------------------------------------------------------
print(f"\nbest loss = {best_loss:.2e}   (noise floor {NOISE**2:.1e})")
print(f"max |err|   A = {(A_f - A_t).abs().max().item():.3f}   "
      f"B = {(B_f - B_t).abs().max().item():.3f}   "
      f"c = {(c_f - c_t).abs().max().item():.3f}   "
      f"d = {abs(d_f.item() - d_t):.3f}")
torch.set_printoptions(precision=3, sci_mode=False)
print("\nA true:\n", A_t)
print("A recovered:\n", A_f)

# ---- 6. fitted outputs + the recovered HIDDEN states ----------------------------
Z_fit, Y_fit = mixed_forward(t_sim, A_f, B_f, c_f, d_f)
# ... 2×2 plot: measured vs fitted outputs (top row) and
# true vs recovered hidden states (bottom row), one column per experiment
```

**✏️ Where to change things:**

| What | Where |
|---|---|
| Number of states / outputs | `N = 3` |
| Number of experiments (initial states) | `E = 2` |
| Measurement noise | `NOISE = 0.01` |
| Where & how often outputs are observed | `t_obs = torch.linspace(0, 10, 51)` |
| The true system (your own `A`, `B`, gains `c`, offset `d`) | the block after `# ---- 1. the true system` — replace `A_t, B_t, c_t, d_t` with your own tensors (keep `A_t` stable) |
| Known initial states | `Z0` — one row per experiment; must be known a priori |
| Optimisation effort | `range(8)` restarts and `range(500)` iterations in section 4 |
| Learning rate | `lr=0.05` |

**Using real data instead of a synthetic system:** replace the `Y_obs = ...` line
with your measurements — shape `(E, len(t_obs), N)`, one experiment per row — and
set `Z0` to the initial conditions those experiments actually started from. Then
comment out the `A_t / B_t / c_t / d_t` truth comparison (sections 5–6) — the fit
itself (sections 3–4) needs nothing else. Keep in mind the model assumes each
output channel is its own state times a gain plus a shared offset; if your data
needs full output mixing (`y = C·z + d` with a general matrix C), you'll need more
experiments (E ≥ N) and the identification gets harder.

**How the fit works (why two optimisers):**

* `c, d` enter the outputs **linearly** → `fit_cd` solves them exactly by least
  squares on every step — no gradient descent for them (PART I's philosophy).
* `A, B` enter the ODE **nonlinearly** → Adam backpropagates through the RK4
  solver (PART II · 3's philosophy). This hybrid is called *variable projection*.
* The loss landscape has local minima, so the code restarts 8 times from
  different random guesses and keeps the run with the lowest loss.

**Reading the output:**

```
best loss = 8.79e-05   (noise floor 1.0e-04)
max |err|   A = 0.071   B = 0.042   c = 0.005   d = 0.000
```

* The **best loss should sit at or near the noise floor** `NOISE²` — here 8.8e-5
  vs 1e-4. That is the single most important check: it means the fitted outputs
  explain the data as well as the truth can.
* The `max |err|` line compares recovered parameters with the (known) truth.
  Values of ~0.01–0.1 are excellent; individual `A` entries carry more
  uncertainty than `c`/`d` because the state is hidden.
* The bottom-row plots show the **hidden state recovered from outputs alone** —
  solid = true, dashed = recovered.

---

## 4. Expected training times

Measured on this machine (Windows 11, CPU-only torch 2.13.0):

| Section | Work | Time |
|---|---|---|
| PART I — joint regression | 1 sklearn fit, ~30 samples | **< 1 s** |
| PART II · 1 — scalar ODE check | 3 RK4/Euler solves of 101 steps | **< 1 s** |
| PART II · 2 — N-dim system check | 3 solves (121 + 1201 steps) | **1–2 s** |
| PART II · 3 — fit `p, a` | 400 Adam steps × 26-step RK4 | **a few seconds** |
| PART II · 4 — fit `A, B, c, d` | 8 restarts × 500 Adam steps | **~4 min total** (~30 s per restart, measured) |

**Speed tips for PART II · 4** (time scales linearly with the bold items):

* Fewer restarts: `range(8)` → `range(4)` halves the time (a restart typically
  finishes in ~30 s; on the default seeds start #1 already reaches the noise floor).
* Fewer iterations: `range(500)` → `range(300)` — successful runs converge
  by iteration ~200–400.
* Shorter / sparser `t_obs`: 51 → 31 points.
* If you later install a CUDA build of torch, move the tensors to GPU — but for
  a problem this small the gain is modest.

> If you *only* want the plots and solver checks (PART II · 1–2), there is no
> training at all — everything is instant.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| PART II · 4 best loss stays ≫ noise floor (e.g. 1e-3 vs 1e-4) | A restart landed in a local minimum — raise the restart count (`range(8)` → 12) or change the seeds (`torch.manual_seed(101/201)`). The loss-vs-floor gap is the reliable quality signal. |
| PART I prints `Expected 6 unknowns but found 5` | An equation uses a name that isn't shared/declared — check spelling and `EXPECTED_UNKNOWNS`. |
| PART I warns an equation is non-linear in its parameters | e.g. `a*exp(b*x)` — switch to `scipy.optimize.curve_fit` for that equation. |
| Trajectories blow up when you hand-write `A` | Your `A` is unstable (`Re(λ) ≥ 0`). Use `make_stable_matrix` or shift `A - s·I`. |
| `UnicodeEncodeError` if you run the code outside Jupyter on Windows | Console is cp1252 — run with `PYTHONIOENCODING=utf-8`, or just use Jupyter (it's UTF-8). |
| `No module named 'torch'` | `pip install torch` (CPU build is fine) — or activate the repo's `.venv`. |
| 3-D phase portrait doesn't render | `pip install matplotlib` ≥ 3.4, or set `N_STATES = 2` for the 2-D portrait. |

---

## 6. Cheat sheet — the whole workflow

1. **Setup once:** `pip install numpy pandas scikit-learn matplotlib torch` (or use `.venv`).
2. **PART I — your equations & data:** edit `EQUATIONS` (section 2.1) and
   `DATA_SOURCES` (section 2.2) → Run All → read fitted parameters + R²/RMSE. *Seconds.*
3. **PART II · 2 — your own ODE system:** edit `N_STATES`, `N_INPUTS`, and the
   `A`, `B`, `z0` block (section 3.2) → check RK4 error is tiny. *Seconds.*
4. **PART II · 4 — fit the hidden system:** edit `N`, `E`, `NOISE`, `t_obs` and
   the true-system block (section 3.4) → Run → confirm `best loss ≈ NOISE²` →
   read the recovered-vs-true table and plots. *~4 min on CPU (8 restarts × 30 s).*
