# This cell creates a ready-to-run Jupyter/Colab notebook (.ipynb)
# that implements the full F–α–ε–P framework as a reusable blueprint.
# The notebook includes explanations, code, calculations, and visualizations.
import nbformat as nbf
from textwrap import dedent
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# 0) Title & Overview
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
# **F–α–ε–P Framework Blueprint (Qualitativ hohes methodisch‑empirisches Werkzeug)**
**A reusable Colab notebook** to evaluate models, experiments, or processes under heavy tails, sequential updates, and uncertainty—complete with explanations, code, calculations, and visualizations.

> **What you get**
> - **F (Concretion)**: measurable outcomes & empirical distribution \(F\).
> - **α (Ingression)**: right/left tail indices via Hill-type estimators (heavy-tail sensitivity).
> - **ε (Involution)**: quantile-based truncation & sequential update coherence.
> - **P (Projection/Decision)**: scoring aggregator using quadratic / harmonic / geometric **quasinorm lenses**.
> - **Visualizations**: distribution, QQ plots, lens scores, streaming updates, and the 3D evaluation cube.
>
> **How to use**
> 1. Run the cells top-to-bottom.
> 2. Either **simulate data** (default) or **load your own CSV** (instructions below).
> 3. Adjust the tolerance \( \varepsilon \) and the lens weights \((\alpha_2,\alpha_{-1},\alpha_0)\).
> 4. Read the inline commentary to transfer this to any formalized evaluation context.
""")))

# 1) Imports & Config
cells.append(nbf.v4.new_code_cell(dedent(r"""
# === Imports ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt, exp
from pathlib import Path

# (No seaborn; one plot per figure; no explicit color choices.)
np.random.seed(42)

# Display options
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
""")))

# 2) Data Ingestion (simulate or load)
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## 1) Data Ingestion (F – Concretion)
You can either **simulate** a dataset with heavy tails or **load your own**:

- **Simulated**: mixture of light- and heavy-tailed components with optional multiplicative drift.
- **Own data**: provide a CSV path and column name for the measurement/metric/residuals.
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
# === OPTION A: Simulated dataset ===
n = 5000
# Light-tailed component (normal)
x_light = np.random.normal(loc=0.0, scale=1.0, size=n)
# Heavy-tailed component (Pareto-like, shifted)
# Pareto(alpha) with minimum x_m=1 => draw U, X = x_m * (1-U)^(-1/alpha)
alpha_true = 3.0
U = np.random.rand(n)
x_heavy = (1 - U) ** (-1.0 / alpha_true)  # Pareto tail
x_heavy = x_heavy - np.mean(x_heavy)      # center-ish

# Mix & add multiplicative drift
mix = np.where(np.random.rand(n) < 0.8, x_light, x_heavy)
# Construct a "residual-like" series with drift in magnitude
t = np.arange(n)
drift = np.exp(0.001 * (t - n/2))  # mild multiplicative drift
r_sim = mix * drift

df = pd.DataFrame({"r": r_sim})
df.head()
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
# === OPTION B: Load your own CSV ===
# Uncomment and set the path & column if you want to use your data.
# csv_path = "/content/your_data.csv"
# value_column = "your_metric_column"
# df = pd.read_csv(csv_path)
# df = df[[value_column]].rename(columns={value_column: "r"})
# df.head()
""")))

# 3) Empirical Distribution F, Quantiles, Epsilon thresholds
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## 2) Empirical Distribution \(F\) and ε-Quantile Truncation (ε – Involution)
We compute the empirical CDF \(F\) through quantiles and select **upper** and **lower** truncation thresholds:
- \( u_\varepsilon = F^{-1}(1-\varepsilon) \)
- \( \ell_\varepsilon = F^{-1}(\varepsilon) \)

This splits the sample into **bulk** and **tails** to guarantee \(\varepsilon\)-controlled error on expectation-like functionals.
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
# Choose epsilon (tolerance)
epsilon = 0.01  # 1% tails on each side by default

abs_r = np.abs(df["r"].values)
u_epsilon = np.quantile(abs_r, 1 - epsilon)  # upper cutoff
l_epsilon = np.quantile(abs_r, epsilon)      # lower cutoff (near zero)

bulk_mask = (abs_r >= l_epsilon) & (abs_r <= u_epsilon)
tail_hi_mask = abs_r > u_epsilon
tail_lo_mask = abs_r < l_epsilon

n_bulk = bulk_mask.sum()
n_hi = tail_hi_mask.sum()
n_lo = tail_lo_mask.sum()

print(f"epsilon = {epsilon}")
print(f"Upper cutoff u_epsilon = {u_epsilon:.4f}, Lower cutoff l_epsilon = {l_epsilon:.6f}")
print(f"Counts -> bulk: {n_bulk}, upper tail: {n_hi}, lower 'near-zero' tail: {n_lo}")
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
# Visualize distribution and cutoffs
fig = plt.figure(figsize=(6,4))
_ = plt.hist(abs_r, bins=60)
_ = plt.axvline(u_epsilon, linestyle="--")
_ = plt.axvline(l_epsilon, linestyle="--")
_ = plt.title("Distribution of |r| with ε-cutoffs")
_ = plt.xlabel("|r|")
_ = plt.ylabel("Count")
plt.show()
""")))

# 4) Tail Indices (alpha, beta) via Hill-type estimators
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## 3) Tail Indices (α, β) – Ingression
Estimate heavy-tail behavior with Hill-type estimators:

- **Right tail α** (large values): use top-k order statistics of \(|r|\).
- **Left tail β** (small values): transform \(y_i = 1/|r_i|\) and apply Hill to \(y\) (right tail of \(y\) corresponds to left tail of \(|r|\)).

> Note: Choose \(k\) sensibly (e.g., 1–10% of data). We provide a simple heuristic and a basic stability plot.
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
def hill_estimator(x_sorted_desc, k):
    \"\"\"
    Basic Hill estimator on descending-sorted positive array x.
    Returns alpha_hat (tail index) for Pareto-like tail.
    \"\"\"
    x = x_sorted_desc
    if k <= 1 or k >= len(x):
        return np.nan
    xk = x[:k]
    xk_min = x[k]  # threshold (k+1-th largest)
    logs = np.log(xk) - np.log(xk_min)
    hill = 1.0 / (np.mean(logs) + 1e-12)
    return hill

# Prepare data for tails
x_pos = abs_r[abs_r > 0]
x_sorted = np.sort(x_pos)
x_sorted_desc = x_sorted[::-1]

# Heuristic k: top ~2% (at least 50 points)
k_right = max(int(0.02 * len(x_sorted_desc)), 50)
alpha_hat = hill_estimator(x_sorted_desc, k_right)

# Left tail via y=1/|r| (large y => small |r|)
y = 1.0 / x_pos
y_sorted = np.sort(y)
y_sorted_desc = y_sorted[::-1]
k_left = max(int(0.02 * len(y_sorted_desc)), 50)
beta_hat = hill_estimator(y_sorted_desc, k_left)

print(f"Hill tail index estimates: alpha (right tail) ≈ {alpha_hat:.3f}, beta (left/near-zero) ≈ {beta_hat:.3f}")
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
# Stability scan for k
k_vals = np.unique(np.linspace(30, max(60, int(0.1*len(x_sorted_desc))), 12, dtype=int))
alpha_series = [hill_estimator(x_sorted_desc, k) for k in k_vals]
beta_series  = [hill_estimator(y_sorted_desc, k) for k in k_vals]

fig = plt.figure(figsize=(6,4))
_ = plt.plot(k_vals, alpha_series, marker="o")
_ = plt.axhline(alpha_hat, linestyle="--")
_ = plt.title("Right-tail Hill α vs k")
_ = plt.xlabel("k (top order stats)")
_ = plt.ylabel("α estimate")
plt.show()

fig = plt.figure(figsize=(6,4))
_ = plt.plot(k_vals, beta_series, marker="o")
_ = plt.axhline(beta_hat, linestyle="--")
_ = plt.title("Left-tail Hill β vs k (via y=1/|r|)")
_ = plt.xlabel("k (top order stats of y)")
_ = plt.ylabel("β estimate")
plt.show()
""")))

# 5) Quasinorm Lenses (L2, harmonic, geometric)
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## 4) Quasinorm Lenses (Quadratic L², Harmonic L⁻¹, Geometric L⁰)
Compute on **bulk** (ε-truncated) data to guarantee finite, stable contributions.

- \( \|r\|_{(2)} = \sqrt{\frac{1}{n}\sum r_i^2} \)
- \( \|r\|_{(-1)} = \left(\frac{1}{n}\sum \frac{1}{|r_i|}\right)^{-1} \)
- \( \|r\|_{(0)} = \exp\Big(\frac{1}{n}\sum \log|r_i|\Big) \)
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
def lens_L2(x):
    return np.sqrt(np.mean(x**2))

def lens_Lminus1(x):
    # avoid division by zero in bulk by design; add tiny epsilon safeguard
    eps = 1e-12
    return 1.0 / np.mean(1.0 / (np.abs(x) + eps))

def lens_L0(x):
    eps = 1e-12
    return np.exp(np.mean(np.log(np.abs(x) + eps)))

r_bulk = df["r"].values[bulk_mask]
L2 = lens_L2(r_bulk)
Lminus1 = lens_Lminus1(r_bulk)
L0 = lens_L0(r_bulk)

print(f"L2 (quadratic)     = {L2:.4f}")
print(f"L^{-1} (harmonic)  = {Lminus1:.4f}")
print(f"L^0 (geometric)    = {L0:.4f}")
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
# Visualize lens values
vals = [L2, Lminus1, L0]
labels = ["L2 (quadratic)", "L^{-1} (harmonic)", "L^0 (geometric)"]

fig = plt.figure(figsize=(6,4))
_ = plt.bar(labels, vals)
_ = plt.title("Quasinorm lenses on ε-bulk")
_ = plt.ylabel("value")
plt.xticks(rotation=15)
plt.show()
""")))

# 6) Aggregate score P with weights
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## 5) Projection / Aggregation (P)
Aggregate the lenses with weights \((\alpha_2,\alpha_{-1},\alpha_0)\) to get a **single score** \( \mathcal{E} \).

Choose weights according to context:
- Risk-averse (emphasize extremes): higher weight on L².
- Bottleneck-averse (avoid small values): higher weight on L⁻¹.
- Multiplicative growth consistency: higher weight on L⁰.
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
alpha2, alpha_minus1, alpha0 = 1/3, 1/3, 1/3  # balanced by default
w_sum = alpha2 + alpha_minus1 + alpha0
alpha2, alpha_minus1, alpha0 = alpha2/w_sum, alpha_minus1/w_sum, alpha0/w_sum

E_score = alpha2*L2 + alpha_minus1*Lminus1 + alpha0*L0
print(f"Weights: alpha2={alpha2:.3f}, alpha-1={alpha_minus1:.3f}, alpha0={alpha0:.3f}")
print(f"Aggregate score 𝓔 = {E_score:.4f}")
""")))

# 7) Sequential updates demo
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## 6) Sequential Updates (Involution – coherence over time)
We simulate a **streaming** scenario: compute the three lenses and the aggregate score on a rolling window to show stable update behavior.
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
window = 500
r = df["r"].values
t_vals = []
E_vals, L2_vals, Lm1_vals, L0_vals = [], [], [], []

for end in range(window, len(r)+1, 50):
    seg = r[end-window:end]
    abs_seg = np.abs(seg)
    u = np.quantile(abs_seg, 1 - epsilon)
    l = np.quantile(abs_seg, epsilon)
    bulk = seg[(abs_seg >= l) & (abs_seg <= u)]
    if len(bulk) < 10:
        continue
    l2 = lens_L2(bulk)
    lm1 = lens_Lminus1(bulk)
    l0 = lens_L0(bulk)
    score = alpha2*l2 + alpha_minus1*lm1 + alpha0*l0
    t_vals.append(end)
    L2_vals.append(l2); Lm1_vals.append(lm1); L0_vals.append(l0); E_vals.append(score)

fig = plt.figure(figsize=(6,4))
_ = plt.plot(t_vals, E_vals, marker="o")
_ = plt.title("Aggregate score 𝓔 over time (rolling window)")
_ = plt.xlabel("time index")
_ = plt.ylabel("𝓔")
plt.show()

fig = plt.figure(figsize=(6,4))
_ = plt.plot(t_vals, L2_vals, label="L2")
_ = plt.plot(t_vals, Lm1_vals, label="L^{-1}")
_ = plt.plot(t_vals, L0_vals, label="L^0")
_ = plt.title("Lens values over time")
_ = plt.xlabel("time index")
_ = plt.ylabel("lens value")
_ = plt.legend()
plt.show()
""")))

# 8) 3D Evaluation Cube visualization
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## 7) 3D Evaluation Cube (C, I, V)
A simple 3D illustration: the **necessary corner (1,1,1)** corresponds to fair evaluation (all three axes satisfied). We place our current evaluation there once the components are computed.
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
# Cube drawing similar to earlier, minimalistic
from mpl_toolkits.mplot3d import Axes3D  # noqa

verts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]])
edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
for e in edges:
    pts = verts[list(e)]
    ax.plot(pts[:,0], pts[:,1], pts[:,2])
for v in verts:
    ax.scatter(v[0], v[1], v[2], s=30)
ax.text(1.03,1.03,1.03,"Fair (1,1,1)")
ax.set_xlabel("C (Concretion)")
ax.set_ylabel("I (Ingression)")
ax.set_zlabel("V (Involution)")
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_zticks([0,1])
ax.view_init(elev=20, azim=30)
plt.title("Evaluation Cube")
plt.show()
""")))

# 9) Reporting helper
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## 8) Summary Report
This cell prints a concise, copy-paste ready summary for documentation or decision memos.
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
def summary_report():
    lines = []
    lines.append("=== F–α–ε–P Evaluation Summary ===")
    lines.append(f"epsilon (tolerance): {epsilon}")
    lines.append(f"Right tail index α (Hill): {alpha_hat:.3f}")
    lines.append(f"Left tail index β  (Hill via y=1/|r|): {beta_hat:.3f}")
    lines.append(f"Upper cutoff u_epsilon: {u_epsilon:.6f}")
    lines.append(f"Lower cutoff l_epsilon: {l_epsilon:.6f}")
    lines.append(f"Bulk size: {n_bulk}, Upper tail size: {n_hi}, Lower tail size: {n_lo}")
    lines.append("--- Lenses on ε-bulk ---")
    lines.append(f"L2 (quadratic): {L2:.6f}")
    lines.append(f"L^-1 (harmonic): {Lminus1:.6f}")
    lines.append(f"L^0 (geometric): {L0:.6f}")
    lines.append("--- Weights ---")
    lines.append(f"alpha2={alpha2:.3f}, alpha-1={alpha_minus1:.3f}, alpha0={alpha0:.3f}")
    lines.append(f"Aggregate score 𝓔: {E_score:.6f}")
    return "\n".join(lines)

print(summary_report())
""")))

# 10) Packaging as functions / class
cells.append(nbf.v4.new_markdown_cell(dedent(r"""
## 9) Reusable API (apply to any metric/residual column)
Use `evaluate_series(series, epsilon, weights)` on your own data.
""")))

cells.append(nbf.v4.new_code_cell(dedent(r"""
def evaluate_series(series: np.ndarray, epsilon: float = 0.01, weights=(1/3,1/3,1/3)):
    series = np.asarray(series).astype(float)
    abs_s = np.abs(series[series!=0])
    if len(abs_s) < 100:
        raise ValueError("Need at least 100 non-zero points for stable tail/lens estimates.")
    u = np.quantile(abs_s, 1 - epsilon)
    l = np.quantile(abs_s, epsilon)
    bulk = series[(np.abs(series) >= l) & (np.abs(series) <= u)]
    # Hill estimates
    x_sorted_desc = np.sort(abs_s)[::-1]
    y_sorted_desc = np.sort(1.0/abs_s)[::-1]
    k_right = max(int(0.02 * len(x_sorted_desc)), 50)
    k_left  = max(int(0.02 * len(y_sorted_desc)), 50)
    a_hat = hill_estimator(x_sorted_desc, k_right)
    b_hat = hill_estimator(y_sorted_desc, k_left)
    # lenses
    L2v = lens_L2(bulk)
    Lm1v = lens_Lminus1(bulk)
    L0v = lens_L0(bulk)
    w = np.array(weights, dtype=float)
    w = w / np.sum(w)
    E = w[0]*L2v + w[1]*Lm1v + w[2]*L0v
    return {
        "epsilon": epsilon,
        "u_epsilon": float(u),
        "l_epsilon": float(l),
        "alpha_hat": float(a_hat),
        "beta_hat": float(b_hat),
        "L2": float(L2v),
        "L-1": float(Lm1v),
        "L0": float(L0v),
        "weights": w.tolist(),
        "E": float(E),
        "n_bulk": int(len(bulk))
    }

# Example usage:
res = evaluate_series(df["r"].values, epsilon=0.01, weights=(0.5,0.25,0.25))
res
""")))

# 11) Save notebook to file
nb["cells"] = cells
out_path = Path("C:/Users/Dominik/Downloads/civ/F_alpha_epsilon_P_framework_blueprint.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

str(out_path)
