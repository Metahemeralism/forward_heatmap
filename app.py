"""Streamlit UI — electricity forward/futures pricer.

Layout mirrors a Black-Scholes option heatmap (reference screenshot):
  * Left sidebar : all model parameters + heatmap axis bounds
  * Main area    : headline cards (Spot, 1Y Forward), forward curve, heatmap,
                   and a sample-path simulator to visualise what the parameters
                   produce.

The underlying model is in models.py; Monte Carlo in simulation.py. Read those
first if you want the maths — this file is concerned only with wiring the UI
and sanity-checking inputs.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from models import JumpParams, LuciaSchwartzJumpModel, SeasonalityParams
from simulation import simulate_paths


# ----------------------------------------------------------------------
# Page config + matplotlib theming (dark, to match the reference aesthetic)
# ----------------------------------------------------------------------
st.set_page_config(page_title="Electricity Forward Pricer", layout="wide")
# We drive all matplotlib theming through an explicit rcParams dict so legend
# text, tick labels, grid, etc. are all readable on Streamlit's dark background.
# Seaborn's style= presets mutate some of these under the hood, so we set our
# own values AFTER calling set_theme to make sure they stick.
sns.set_theme(style="dark")
plt.rcParams.update({
    # Backgrounds
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#0e1117",
    "savefig.facecolor": "#0e1117",
    # Borders and grid
    "axes.edgecolor":  "#3a4150",
    "grid.color":      "#2c3340",
    "grid.alpha":      0.5,
    # All text on the plot — high contrast against the dark background
    "text.color":        "#fafafa",
    "axes.labelcolor":   "#fafafa",
    "axes.titlecolor":   "#fafafa",
    "xtick.color":       "#fafafa",
    "ytick.color":       "#fafafa",
    # Legend: readable box with high-contrast label text
    "legend.facecolor":  "#1a1f2b",
    "legend.edgecolor":  "#3a4150",
    "legend.labelcolor": "#fafafa",
})


def dollar(x: float, places: int = 2) -> str:
    """Format a currency value while escaping the '$' for Streamlit markdown.

    Streamlit treats paired '$' characters as LaTeX math delimiters, so a bare
    f-string like f"${x:.2f}" would show up as rendered math (and look green on
    the dark theme). Escaping with '\\$' keeps it as literal text.
    """
    return f"\\${x:,.{places}f}"


# ----------------------------------------------------------------------
# Sidebar: model parameters
# ----------------------------------------------------------------------
st.sidebar.title("Model Parameters")

st.sidebar.markdown("#### Spot & short-term factor")
S0 = st.sidebar.number_input(
    "Spot price  S₀  ($/MWh)", min_value=0.10, value=50.00, step=1.0, format="%.2f",
)
kappa = st.sidebar.number_input(
    "Mean-reversion speed  κ  (per year)", min_value=0.01, value=1.50, step=0.10, format="%.2f",
)
sigma_X = st.sidebar.number_input(
    "Short-term volatility  σ_X", min_value=0.0, value=0.80, step=0.05, format="%.2f",
)
X0 = st.sidebar.number_input(
    "Initial short deviation  X₀  (log)", value=0.0, step=0.05, format="%.2f",
)

st.sidebar.markdown("#### Long-term factor")
sigma_Y = st.sidebar.number_input(
    "Long-term volatility  σ_Y", min_value=0.0, value=0.10, step=0.01, format="%.2f",
)
mu_Y = st.sidebar.number_input(
    "Long-term drift  μ_Y  (log/year)", value=0.02, step=0.01, format="%.3f",
)
rho = st.sidebar.slider(
    "Correlation  ρ(W_X, W_Y)", min_value=-1.0, max_value=1.0, value=0.0, step=0.05,
)

st.sidebar.markdown("#### Seasonality (annual harmonic)")
season_amp = st.sidebar.number_input(
    "Amplitude (log-space)", min_value=0.0, value=0.20, step=0.01, format="%.2f",
)
season_phase = st.sidebar.slider(
    "Peak month offset (years from t=0)", min_value=0.0, max_value=1.0, value=0.0, step=1 / 12,
)

st.sidebar.markdown("#### Jumps (Merton-style)")
jump_intensity = st.sidebar.number_input(
    "Intensity  λ  (jumps/year)", min_value=0.0, value=4.0, step=0.5, format="%.2f",
)
jump_mean = st.sidebar.number_input(
    "Mean log-jump  μ_J", value=0.15, step=0.05, format="%.2f",
)
jump_std = st.sidebar.number_input(
    "Std log-jump  σ_J", min_value=0.0, value=0.30, step=0.05, format="%.2f",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Heatmap axes")
T_min = st.sidebar.number_input(
    "Min time-to-delivery (years)", min_value=0.01, value=0.05, step=0.05, format="%.2f",
)
T_max = st.sidebar.number_input(
    "Max time-to-delivery (years)", min_value=0.10, value=2.00, step=0.25, format="%.2f",
)
sigX_min = st.sidebar.slider(
    "Min σ_X (heatmap axis)", min_value=0.0, max_value=2.0, value=0.20, step=0.05,
)
sigX_max = st.sidebar.slider(
    "Max σ_X (heatmap axis)", min_value=0.1, max_value=3.0, value=1.50, step=0.05,
)

# ----------------------------------------------------------------------
# Guard-rail: a few input combinations are silly and will look weird downstream.
# Better to flag them upfront than to render a nonsense plot.
# ----------------------------------------------------------------------
if T_max <= T_min:
    st.error("Max time-to-delivery must be greater than min. Adjust the sidebar.")
    st.stop()
if sigX_max <= sigX_min:
    st.error("Max σ_X must be greater than min. Adjust the sidebar.")
    st.stop()


# ----------------------------------------------------------------------
# Build the model
# ----------------------------------------------------------------------
seasonality = SeasonalityParams(amplitude=season_amp, phase_years=season_phase)
jumps = JumpParams(intensity=jump_intensity, mean_log=jump_mean, std_log=jump_std)
model = LuciaSchwartzJumpModel(
    S0=S0, kappa=kappa, sigma_X=sigma_X, sigma_Y=sigma_Y, rho=rho, mu_Y=mu_Y,
    X0=X0, seasonality=seasonality, jumps=jumps,
)

# ----------------------------------------------------------------------
# Headline cards: Spot  and  1-Year Forward
# Mirrors the "Call / Put" cards in the reference screenshot.
# ----------------------------------------------------------------------
F_1y = model.forward_price(1.0)

card_style = (
    "padding:14px;border-radius:8px;text-align:center;"
    "font-size:22px;color:#111;font-weight:500"
)
c1, c2 = st.columns(2)
c1.markdown(
    f"<div style='background-color:#a8e6a3;{card_style}'>"
    f"<b>Spot Price</b><br>${S0:,.2f}/MWh</div>",
    unsafe_allow_html=True,
)
c2.markdown(
    f"<div style='background-color:#f8c0c0;{card_style}'>"
    f"<b>1-Year Forward</b><br>${F_1y:,.2f}/MWh</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Title + teaching intro
# ----------------------------------------------------------------------
st.title("Electricity Forward / Futures Pricer")
st.markdown(
    "Pricing under the **Lucia-Schwartz two-factor log-price model** with "
    "**Merton-style jumps** on the short-term factor. Forward prices are "
    "evaluated in closed form under the risk-neutral measure Q."
)

with st.expander("How is this model different from Black-Scholes?"):
    st.markdown(
        """
**Black-Scholes assumes a geometric Brownian motion** — prices drift and diffuse
with ever-growing variance. That is a reasonable approximation for stocks but a
poor one for electricity:

1. **Mean reversion** — Power is (mostly) non-storable, so prices revert toward
   the marginal cost of generation. We model this with an Ornstein-Uhlenbeck
   short-term factor $X_t$.
2. **Seasonality** — Annual demand patterns drive deterministic swings. A
   cosine term $f(t)$ handles this.
3. **Spikes** — Grid stress, outages, and weather trigger brief price
   explosions. A compound-Poisson jump component on $X_t$ captures these.
4. **Long-run drift** — Technology + carbon prices + capacity drift over years.
   A second factor $Y_t$ (arithmetic Brownian motion) captures this.

Putting it together:
$$\\ln S_t = f(t) + X_t + Y_t$$

with  $dX_t = -\\kappa X_t\\, dt + \\sigma_X\\, dW_X + J\\, dN$  (mean-reversion + jumps)
and   $dY_t = \\mu_Y\\, dt + \\sigma_Y\\, dW_Y$  (long-run drift).
"""
    )


# ----------------------------------------------------------------------
# Panel 1: forward curve F(0, T) as a function of delivery time T
# ----------------------------------------------------------------------
st.subheader("Forward Curve  F(0, T)")
T_grid = np.linspace(T_min, T_max, 250)          # shape: (250,)
F_grid = model.forward_curve(T_grid)             # shape: (250,)

fig_curve, ax_curve = plt.subplots(figsize=(10, 3.6))
ax_curve.plot(T_grid, F_grid, color="#4cc9f0", lw=2.0, label="F(0, T)")
ax_curve.axhline(S0, color="#f8c0c0", lw=1, ls="--", label="Spot S₀")
ax_curve.set_xlabel("Time to delivery  T  (years)")
ax_curve.set_ylabel("Forward price  ($/MWh)")
ax_curve.grid(alpha=0.2)
ax_curve.legend(loc="best")
st.pyplot(fig_curve)

with st.expander("Reading the forward curve"):
    st.markdown(
        """
Each point on this curve is the **fair price today of a contract that delivers 1 MWh
of power at time $T$**. Three features to look for:

- **Shape vs. spot.** If the curve is above spot, the market is in *contango*
  (long-term expectations exceed current spot); below spot is *backwardation*.
- **Seasonality.** You should see the annual cosine ripple. Adjust the
  seasonality amplitude / phase in the sidebar and watch it appear/move.
- **Jump convexity.** Turning up λ or $\\sigma_J$ lifts the curve even with
  zero long-run drift — that's the Merton convexity correction:
  $\\mathbb{E}[e^J] > e^{\\mathbb{E}[J]}$ when J is Gaussian, because the
  exponential is convex.
"""
    )


# ----------------------------------------------------------------------
# Panel 2: HEATMAP — F(0, T) across (T, sigma_X). This is the direct
# analogue of the reference call/put price heatmap (strike/vol replaced by
# time-to-delivery/short-term-vol).
# ----------------------------------------------------------------------
st.subheader("Forward Price Heatmap  —  Time-to-delivery  ×  Short-term volatility")

T_axis = np.linspace(T_min, T_max, 10)                      # shape: (10,)
sig_axis = np.linspace(sigX_min, sigX_max, 10)              # shape: (10,)

# Build a (10, 10) price grid: rows = sigma_X, cols = T.
# We rebuild the model for each sigma_X (cheap — just a dataclass) so the
# closed-form forward uses the updated vol. This isn't the most elegant code,
# but it keeps the model itself immutable, which is easier to reason about.
grid = np.empty((len(sig_axis), len(T_axis)))
for i, sX in enumerate(sig_axis):
    m_i = LuciaSchwartzJumpModel(
        S0=S0, kappa=kappa, sigma_X=sX, sigma_Y=sigma_Y, rho=rho, mu_Y=mu_Y,
        X0=X0, seasonality=seasonality, jumps=jumps,
    )
    grid[i, :] = m_i.forward_curve(T_axis)                   # row-fill: shape (10,)

fig_hm, ax_hm = plt.subplots(figsize=(10, 5))
sns.heatmap(
    grid,
    xticklabels=[f"{t:.2f}" for t in T_axis],
    yticklabels=[f"{s:.2f}" for s in sig_axis],
    annot=True, fmt=".2f", cmap="viridis",
    ax=ax_hm, cbar_kws={"label": "F(0, T)  ($/MWh)"},
)
ax_hm.set_xlabel("Time to delivery  T  (years)")
ax_hm.set_ylabel("Short-term volatility  σ_X")
st.pyplot(fig_hm)

with st.expander("Why does the heatmap look the way it does?"):
    st.markdown(
        """
Each cell is $F(0, T)$ recomputed with the given $(T, \\sigma_X)$.

- **Moving right (T ↑)** — two competing effects: seasonality cycles the curve
  up and down, and long-run drift + diffusion convexity from $Y_t$ push it
  gradually upward.
- **Moving up ($\\sigma_X$ ↑)** — prices rise because of **diffusion convexity**:
  the forward adds $\\tfrac{1}{2}\\sigma_X^2 / (2\\kappa) \\cdot (1 - e^{-2\\kappa T})$
  to the log-forward. More vol → larger convexity correction. Notice how the
  increase **saturates** at long $T$: mean reversion caps how much the OU
  variance can grow.
- **The jump component** adds a separate lift that is roughly independent of
  $\\sigma_X$ (only depends on κ, λ, μ_J, σ_J).
"""
    )


# ----------------------------------------------------------------------
# Panel 3: Monte Carlo sample paths (with closed-form forward overlaid as
# a sanity check — the MC mean should converge to F(0, T) if everything is
# implemented correctly).
# ----------------------------------------------------------------------
st.subheader("Sample spot-price paths  +  forward-curve sanity check")

with st.expander("Show Monte Carlo paths", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    n_paths = col_a.slider("Number of paths", min_value=20, max_value=2000, value=300, step=20)
    horizon = col_b.slider("Horizon (years)", min_value=0.25, max_value=3.0, value=1.5, step=0.25)
    seed = col_c.number_input("Random seed", value=42, step=1)

    times, S, X_paths, Y_paths = simulate_paths(
        model, T_horizon=float(horizon), n_paths=int(n_paths), seed=int(seed),
    )
    # Shapes:
    #   times   -> (n_steps+1,)
    #   S       -> (n_paths, n_steps+1)

    F_times = model.forward_curve(times)          # shape: (n_steps+1,)
    mc_mean = S.mean(axis=0)                      # shape: (n_steps+1,)

    fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
    # Plot at most 80 paths (otherwise the alpha-blended cloud saturates).
    for p in range(min(80, S.shape[0])):
        ax_mc.plot(times, S[p], color="#4cc9f0", alpha=0.12, lw=0.9)
    ax_mc.plot(times, mc_mean, color="#fca311", lw=2.2, label=f"MC mean  (n={n_paths})")
    ax_mc.plot(times, F_times, color="white", lw=2.2, ls="--", label="Closed-form  F(0, T)")
    ax_mc.set_xlabel("Time (years)")
    ax_mc.set_ylabel("Spot price  ($/MWh)")
    ax_mc.legend(loc="best")
    ax_mc.grid(alpha=0.2)
    st.pyplot(fig_mc)

    # Quantify the MC/closed-form gap at T_horizon to flag implementation bugs.
    gap = mc_mean[-1] - F_times[-1]
    rel_gap = gap / F_times[-1]
    st.caption(
        f"At T = {horizon:.2f} y:  MC mean = {dollar(mc_mean[-1])},  "
        f"closed-form F(0,T) = {dollar(F_times[-1])},  "
        f"absolute gap = {dollar(gap, 3)}  ({rel_gap:+.2%}). "
        "With enough paths the gap shrinks toward zero — if it doesn't, there's a bug."
    )

with st.expander("Why simulate paths if we already have a closed form?"):
    st.markdown(
        """
Two reasons:

1. **Intuition.** Parameters on their own are abstract. Seeing jump arrivals,
   reversion, and the seasonal ripple play out makes the model concrete.
2. **Validation.** Every closed-form formula is a chance for a bug. If the MC
   mean converges to $F(0, T)$ across many parameter settings, we have
   independent evidence that both the formula and the simulator are correct.
   If they disagree, at least one of them is wrong.

Monte Carlo also lets you price things that *don't* have closed forms — Asian
options on electricity, swing options, storage valuation — once you trust the
simulator.
"""
    )
