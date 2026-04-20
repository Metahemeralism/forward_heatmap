"""Monte Carlo simulator for the Lucia-Schwartz + jumps electricity model.

Purposes
--------
  1. Visualise spot-price paths so the user can SEE what the parameters produce
     (spikes, mean reversion, seasonal cycle).
  2. Sanity-check the closed-form forward curve: the MC sample mean of S_T should
     converge to F(0, T) from models.forward_price. If it doesn't, something in
     either the formula or the simulator is wrong.

Discretisation choices
----------------------
  * Short factor X: EXACT Ornstein-Uhlenbeck step. Given X_t, the conditional
    distribution of X_{t+dt} (ignoring jumps) is Gaussian with
        mean  = X_t * exp(-kappa*dt)
        var   = sigma_X^2 / (2 kappa) * (1 - exp(-2 kappa dt))
    Using this instead of Euler-Maruyama eliminates the O(dt) drift bias that
    Euler introduces for OU, which matters for small kappa*dt.
  * Long factor Y: exact (it's arithmetic Brownian motion — Euler IS exact here).
  * Jumps: "lump at end of step" approximation. Count jumps in (t, t+dt] via
    Poisson(lambda*dt) and add their total to X_{t+dt}. This ignores the fact
    that a jump arriving mid-step would itself mean-revert over the remainder
    of the step — a negligible effect when kappa*dt is small.
"""
from __future__ import annotations

import numpy as np

from models import LuciaSchwartzJumpModel


def simulate_paths(model: LuciaSchwartzJumpModel,
                   T_horizon: float,
                   n_paths: int = 200,
                   n_steps_per_year: int = 365,
                   seed: int | None = None):
    """Simulate spot-price paths under Q.

    Parameters
    ----------
    model : LuciaSchwartzJumpModel
        The calibrated model to simulate.
    T_horizon : float
        Horizon in years.
    n_paths : int
        Number of Monte Carlo paths.
    n_steps_per_year : int
        Time-grid resolution. 365 ~= daily steps, fine enough for most parameter
        regimes. Increase if kappa is very large (fast mean reversion needs
        finer dt) or jump intensity is very high.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    times : ndarray of shape (n_steps+1,)
        Grid of times in years, from 0 to T_horizon inclusive.
    S : ndarray of shape (n_paths, n_steps+1)
        Simulated spot prices. S[:, 0] = model.S0 for all paths.
    X : ndarray of shape (n_paths, n_steps+1)
        Short-term factor paths.
    Y : ndarray of shape (n_paths, n_steps+1)
        Long-term factor paths.
    """
    rng = np.random.default_rng(seed)

    # Choose a step count that gives roughly n_steps_per_year. Clip at >= 1.
    n_steps = max(int(np.ceil(T_horizon * n_steps_per_year)), 1)
    dt = T_horizon / n_steps
    times = np.linspace(0.0, T_horizon, n_steps + 1)  # shape: (n_steps+1,)

    # Unpack parameters for speed / readability inside the loop.
    kappa = model.kappa
    sX = model.sigma_X
    sY = model.sigma_Y
    rho = model.rho
    mu_Y = model.mu_Y
    lam = model.jumps.intensity
    muJ = model.jumps.mean_log
    sJ = model.jumps.std_log

    # Precompute exact-OU step coefficients (constants for all steps since dt is fixed).
    ou_decay = np.exp(-kappa * dt)
    # Guard against numerical issue if kappa is tiny: the exact-OU std formula
    # -> sigma_X * sqrt(dt) as kappa -> 0. The closed form is still well-behaved
    # for reasonable kappa > 0 so we use it directly.
    ou_std = sX * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))

    # For correlating dW_Y with dW_X via the shared shock z1 plus an independent z2:
    #   dW_X = sqrt(dt) * z1
    #   dW_Y = sqrt(dt) * (rho * z1 + sqrt(1-rho^2) * z2)
    # We reuse z1 for both the OU-exact step AND the Y correlation — same Brownian
    # shock, just different scaling.
    sqrt1mrho2 = np.sqrt(max(1.0 - rho ** 2, 0.0))

    # Allocate arrays. Shape (n_paths, n_steps+1): rows = paths, cols = time.
    X = np.zeros((n_paths, n_steps + 1))
    Y = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = model.X0
    Y[:, 0] = model.Y0

    for i in range(n_steps):
        # Two independent standard-normal draws per path per step.
        z1 = rng.standard_normal(n_paths)   # shape: (n_paths,)
        z2 = rng.standard_normal(n_paths)   # shape: (n_paths,)

        # --- Short factor X: exact-OU diffusion step --------------------
        # X_{t+dt} | X_t  ~ Normal(X_t * ou_decay, ou_std^2)
        X_next = X[:, i] * ou_decay + ou_std * z1

        # --- Compound-Poisson jumps -------------------------------------
        # Number of jumps in (t, t+dt] for each path: Poisson(lambda * dt).
        # Given n_j jumps, the total jump size is Normal(n_j*muJ, n_j*sigma_J^2).
        # (Sum of iid Gaussians is Gaussian with added mean and variance.)
        if lam > 0.0:
            n_jumps = rng.poisson(lam * dt, size=n_paths)        # shape: (n_paths,)
            # rng.standard_normal(n_paths) is the driver; scale by sJ * sqrt(n_jumps)
            # to get the right variance. n_jumps=0 paths contribute 0 automatically.
            jump_sum = (rng.standard_normal(n_paths)
                        * sJ * np.sqrt(n_jumps.astype(float))
                        + n_jumps * muJ)
            X_next = X_next + jump_sum

        X[:, i + 1] = X_next

        # --- Long factor Y: arithmetic Brownian motion (Euler = exact) --
        dW_Y = np.sqrt(dt) * (rho * z1 + sqrt1mrho2 * z2)
        Y[:, i + 1] = Y[:, i] + mu_Y * dt + sY * dW_Y

    # Combine into spot path: ln S_t = f(t) + X_t + Y_t,  S_t = exp(ln S_t).
    seasonal = model.seasonality(times)                 # shape: (n_steps+1,)
    log_S = seasonal[None, :] + X + Y                   # broadcast -> (n_paths, n_steps+1)
    S = np.exp(log_S)

    return times, S, X, Y
