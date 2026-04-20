# Electricity Forward / Futures Pricer

An interactive Streamlit app that prices electricity forward contracts under the
**Lucia-Schwartz two-factor log-price model** with **Merton-style jumps**. The
UI mirrors a Black-Scholes option-heatmap dashboard — sidebar parameters on the
left, headline cards, a forward curve, a time-to-delivery × volatility heatmap,
and a Monte Carlo sanity panel on the right.

---

## Why this model

Electricity prices behave very differently from stock prices. Three stylised
facts drive the model choice:

| Feature | What it captures | Model component |
|---|---|---|
| **Mean reversion** | Prices revert to the marginal cost of generation (power is mostly non-storable) | Short-term factor $X_t$ — Ornstein-Uhlenbeck |
| **Seasonality** | Annual demand / hydro / weather cycles | Deterministic $f(t)$ — annual cosine harmonic |
| **Spikes** | Grid stress, outages, weather | Compound-Poisson jumps on $X_t$ |
| **Long-run drift** | Technology, carbon, capacity changes | Long-term factor $Y_t$ — arithmetic Brownian motion |

Combined:

$$\ln S_t = f(t) + X_t + Y_t$$

$$dX_t = -\kappa X_t\, dt + \sigma_X\, dW_X + J\, dN, \quad dY_t = \mu_Y\, dt + \sigma_Y\, dW_Y$$

with $dW_X\, dW_Y = \rho\, dt$, jumps $J \sim \mathcal{N}(\mu_J, \sigma_J^2)$, and Poisson intensity $\lambda$.

Under the risk-neutral measure Q, the forward price has the closed form:

$$F(0, T) = e^{f(T)} \cdot \underbrace{\exp\!\Big(\mathbb{E}[D] + \tfrac{1}{2}\mathrm{Var}(D)\Big)}_{\text{Gaussian MGF of diffusion part}} \cdot \underbrace{\exp\!\Big(\lambda \int_0^T (\phi_J(e^{-\kappa u}) - 1)\, du\Big)}_{\text{jump MGF}}$$

where $\phi_J(a) = \mathbb{E}[e^{aJ}] = \exp(a \mu_J + \tfrac{1}{2} a^2 \sigma_J^2)$. The jump
integral has no clean closed form when $\kappa > 0$, so it is computed via
`scipy.integrate.quad`.

---

## Features

- **Headline cards** — spot price and 1-year forward side by side.
- **Forward curve** — $F(0, T)$ over a user-specified delivery horizon, with
  the seasonality ripple, mean-reversion decay, and jump convexity all visible.
- **Heatmap** — $F(0, T)$ as a function of time-to-delivery × short-term
  volatility $\sigma_X$. Direct analogue of the Black-Scholes call/put heatmap.
- **Monte Carlo simulator** — sample paths overlaid with the closed-form
  forward curve. The MC mean converges to $F(0, T)$, which validates both the
  formula and the simulator.
- **Teaching expanders** — each panel has a short "how to read this" box
  explaining what's happening and which parameters to vary.

---

## Installation

Built against the local `data-driven` conda environment.

```bash
conda activate data-driven
pip install -r requirements.txt
```

Requirements: `streamlit`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`.

---

## Run

```bash
conda activate data-driven
streamlit run app.py
```

Opens at http://localhost:8501 by default.

---

## File layout

| File | Role |
|---|---|
| [models.py](models.py) | Lucia-Schwartz + jumps model and closed-form forward. Heavily commented with derivations. |
| [simulation.py](simulation.py) | Monte Carlo path generator (exact OU step + Poisson-thinning jumps) for validation and path plots. |
| [app.py](app.py) | Streamlit UI — sidebar inputs, headline cards, forward curve, heatmap, MC panel. |
| [.streamlit/config.toml](.streamlit/config.toml) | Dark theme configuration. |
| [requirements.txt](requirements.txt) | Python dependencies. |

---

## Sanity checks

The model has five internal consistency checks you can reproduce with:

```bash
conda activate data-driven
python -c "
import numpy as np
from models import LuciaSchwartzJumpModel, JumpParams
from simulation import simulate_paths

# 1. F(0,0) == S0 identity
m = LuciaSchwartzJumpModel(S0=50.0)
assert abs(m.forward_price(0.0) - 50.0) < 1e-9

# 2. Shape propagation
assert m.forward_curve(np.linspace(0.1, 2.0, 5)).shape == (5,)

# 3. Closed-form matches MC (no jumps)
m0 = LuciaSchwartzJumpModel(S0=50.0, sigma_X=0.4, sigma_Y=0.05,
                             jumps=JumpParams(intensity=0.0))
_, S, _, _ = simulate_paths(m0, T_horizon=1.0, n_paths=5000, seed=0)
assert abs(S[:, -1].mean() / m0.forward_price(1.0) - 1) < 0.02

# 4. Closed-form matches MC (with jumps)
m1 = LuciaSchwartzJumpModel(S0=50.0, sigma_X=0.4, sigma_Y=0.05)
_, S, _, _ = simulate_paths(m1, T_horizon=1.0, n_paths=20000, seed=0)
assert abs(S[:, -1].mean() / m1.forward_price(1.0) - 1) < 0.03

# 5. log-forward decomposition sums back to log F(0,T)
comps = m1.log_forward_components(1.0)
assert abs(sum(comps.values()) - np.log(m1.forward_price(1.0))) < 1e-9

print('All OK.')
"
```

---

## Parameter cheat sheet

Reasonable ranges to start exploring (tune to your market):

| Parameter | Meaning | Typical range | Effect |
|---|---|---|---|
| `kappa` | Mean-reversion speed (1/year) | 0.5 – 50 | Half-life of shocks = ln(2) / κ |
| `sigma_X` | Short-term vol | 0.3 – 2.0 | Near-term uncertainty; damps at long T |
| `sigma_Y` | Long-term vol | 0.05 – 0.20 | Dominates at long horizons (not damped) |
| `mu_Y` | Long-term drift | 0.00 – 0.05 | Slope of the log-forward curve |
| `rho` | Correlation | −1 – 1 | Coupling between short and long factors |
| `amplitude` | Seasonality (log) | 0.1 – 0.4 | 0.2 ≈ ±20% around trend |
| `lambda` | Jump intensity (1/year) | 2 – 20 | Average jumps per year |
| `mu_J` | Mean log-jump | 0.1 – 0.5 | Positive → up-spikes dominate |
| `sigma_J` | Log-jump std | 0.2 – 0.6 | Jump-size dispersion |

---

## Roadmap / possible extensions

- **Historical calibration** — fit parameters to real day-ahead prices (Elexon
  BMRS for GB, EPEX / Nord Pool elsewhere).
- **European option pricing** — use the same MGF structure for calls/puts on
  $S_T$.
- **Swing options / storage valuation** — Monte Carlo + least-squares Monte
  Carlo (Longstaff-Schwartz) on top of the existing simulator.
- **Weekly harmonic** — add a second seasonal term to capture weekday/weekend.
- **Multi-market spark spreads** — couple this model with a gas price model
  and price a gas-generator's option value.

---

## References

- Lucia, J. J. and Schwartz, E. S. (2002). *Electricity prices and power
  derivatives: Evidence from the Nordic power exchange.* Review of
  Derivatives Research, 5(1), 5–50.
- Cartea, Á. and Figueroa, M. G. (2005). *Pricing in electricity markets: A
  mean reverting jump diffusion model with seasonality.* Applied Mathematical
  Finance, 12(4), 313–335.
- Merton, R. C. (1976). *Option pricing when underlying stock returns are
  discontinuous.* Journal of Financial Economics, 3(1–2), 125–144.
