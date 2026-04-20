"""Lucia-Schwartz two-factor electricity price model with Merton jumps.

MATHEMATICAL BACKGROUND
-----------------------
Electricity prices behave very differently from stock prices. Three stylised facts
drive the model choice:

  1. Mean reversion: power is (mostly) non-storable, so prices revert to the
     marginal cost of generation. A random walk (like Black-Scholes) is the wrong
     long-run behaviour — it lets the variance grow forever.
  2. Seasonality: annual (and weekly) patterns in demand drive deterministic swings.
  3. Spikes: grid stress, outages, and weather cause brief price explosions that
     a pure Gaussian diffusion cannot reproduce.

Lucia & Schwartz (2002) introduced a two-factor model that handles (1) and (2).
We extend it with Merton-style compound-Poisson jumps to handle (3).

THE MODEL
---------
We model the LOG spot price as

    ln S_t = f(t) + X_t + Y_t

where
    f(t) : deterministic seasonality (annual harmonic here)
    X_t  : SHORT-term factor — Ornstein-Uhlenbeck (mean-reverting to 0) with jumps
    Y_t  : LONG-term factor  — arithmetic Brownian motion (captures slow drift)

SDEs under the risk-neutral measure Q (which we price under directly):

    dX_t = -kappa * X_t * dt + sigma_X * dW_X + J * dN       (OU + jumps)
    dY_t =  mu_Y * dt         + sigma_Y * dW_Y                (ABM)
    dW_X dW_Y = rho * dt

Jumps: N_t is Poisson(intensity = lambda). Jump sizes J ~ Normal(mu_J, sigma_J^2),
iid and independent of the Brownian motions.

INTUITION
---------
  * kappa (mean reversion speed): 1/kappa is the "half-life" of a shock.
    Large kappa => shocks decay fast => forward curve flattens quickly with T.
  * sigma_X (short vol): drives near-term uncertainty. Heavily damped at long T
    because of mean reversion (variance saturates at sigma_X^2 / (2 kappa)).
  * sigma_Y (long vol): drives long-run uncertainty. NOT damped — variance grows
    linearly with T. Dominates at long horizons.
  * lambda, mu_J, sigma_J: compound-Poisson jumps. Mean of jump contribution to
    forward: positive if mu_J > 0 (up-spikes), negative otherwise.

FORWARD PRICING
---------------
A forward contract settles on S_T; its fair price is F(t, T) = E^Q_t[S_T].
Because ln S_T is a sum of a Gaussian part (diffusion) and a compound-Poisson
part (jumps), and the two are independent under Q:

    F(0, T) = exp(f(T))  *  G_diff(T)  *  G_jump(T)

where
    G_diff(T) = exp( E[diff] + 0.5 * Var[diff] )     # Gaussian MGF at 1
    G_jump(T) = exp( lambda * integral_0^T [phi_J(e^{-kappa u}) - 1] du )

phi_J(a) = E[exp(a*J)] = exp(a*mu_J + 0.5 * a^2 * sigma_J^2) for Gaussian J.
The factor e^{-kappa u} inside phi_J encodes the mean-reverting decay of a jump
that occurred u years before the delivery date.

The jump integral has no clean closed form when kappa > 0, so we evaluate it
numerically with scipy.integrate.quad. As sanity checks:
  * kappa -> 0: integrand is constant (exp(mu_J + 0.5 sigma_J^2) - 1), so
    G_jump -> exp(lambda * T * (E[e^J] - 1)) — the standard Merton compensator.
  * kappa -> inf: integrand -> 0 for any u > 0, so G_jump -> 1 — jumps decay
    instantly and contribute nothing to the forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad


@dataclass
class SeasonalityParams:
    """Annual harmonic seasonality for log-prices:

        f(t) = amplitude * cos( 2*pi * (t - phase_years) )

    Parameters
    ----------
    amplitude : float
        Peak-to-centre swing in LOG space. For example, amplitude = 0.20 means
        prices peak roughly 22% above the trend (exp(0.20) ~= 1.22) and trough
        roughly 18% below it.
    phase_years : float
        Time (in years from t=0) at which the annual maximum occurs. For a GB
        winter-peaking market, set this to 0.0 if t=0 is 1 January; use 0.5 for
        a summer-peaking market.
    """
    amplitude: float = 0.20
    phase_years: float = 0.0

    def __call__(self, t):
        # Accept scalar or array. Shape out matches shape in.
        t = np.asarray(t, dtype=float)
        return self.amplitude * np.cos(2.0 * np.pi * (t - self.phase_years))


@dataclass
class JumpParams:
    """Merton-style Gaussian jumps in the LOG short-term factor.

    A jump of size J adds J directly to X_t; because we model ln S, this means
    the spot price gets multiplied by exp(J).

    Parameters
    ----------
    intensity : float
        lambda in jumps/year — the expected number of jumps per year.
        Setting this to 0 reduces the model to pure Lucia-Schwartz.
    mean_log : float
        mu_J — mean log-jump size. Positive values bias toward up-spikes
        (typical for electricity, where spikes dominate over crashes).
    std_log : float
        sigma_J — std dev of log-jump size. Controls spike magnitude dispersion.
    """
    intensity: float = 4.0
    mean_log: float = 0.15
    std_log: float = 0.30


@dataclass
class LuciaSchwartzJumpModel:
    """Lucia-Schwartz two-factor model + Merton jumps on the short factor.

    All time inputs are in YEARS. Prices are in arbitrary currency units (we use
    $/MWh in the app). We work directly under the risk-neutral measure Q — the
    user supplies Q-parameters and we trust them.

    Parameters
    ----------
    S0      : spot price at t=0
    kappa   : mean-reversion speed of X (per year). 1/kappa = shock half-life.
    sigma_X : short-term diffusive volatility (per sqrt(year))
    sigma_Y : long-term diffusive volatility (per sqrt(year))
    rho     : correlation between dW_X and dW_Y (in [-1, 1])
    mu_Y    : drift of Y under Q (log-price units per year)
    X0      : initial short-term deviation. Default 0 means "currently at
              equilibrium"; Y0 is then pinned by the spot identity.
    seasonality, jumps : see above.
    """
    S0: float = 50.0
    kappa: float = 1.5
    sigma_X: float = 0.80
    sigma_Y: float = 0.10
    rho: float = 0.0
    mu_Y: float = 0.02
    X0: float = 0.0
    seasonality: SeasonalityParams = field(default_factory=SeasonalityParams)
    jumps: JumpParams = field(default_factory=JumpParams)

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------
    @property
    def Y0(self) -> float:
        """Long-factor initial value, pinned by the identity ln S0 = f(0) + X0 + Y0.

        Rearranging:  Y0 = ln(S0) - f(0) - X0.
        This leaves a single free choice (X0) for splitting the initial log-price
        between short and long factors; we default X0 = 0 (system at equilibrium).
        """
        return float(np.log(self.S0) - self.seasonality(0.0) - self.X0)

    # ------------------------------------------------------------------
    # Gaussian (diffusion-only) moments of ln S_T  given info at t=0
    # ------------------------------------------------------------------
    def _diffusion_mean(self, T):
        # E[X_T^diff] = X0 * e^{-kappa T}                (OU mean decays to 0)
        # E[Y_T]      = Y0 + mu_Y * T                    (ABM mean drifts linearly)
        T = np.asarray(T, dtype=float)
        X_mean = self.X0 * np.exp(-self.kappa * T)
        Y_mean = self.Y0 + self.mu_Y * T
        return X_mean + Y_mean

    def _diffusion_var(self, T):
        # Closed-form variances for OU + ABM with correlation rho:
        #   Var(X_T)      = (sigma_X^2 / (2 kappa)) * (1 - e^{-2 kappa T})
        #                   -> saturates at sigma_X^2 / (2 kappa) as T -> inf
        #   Var(Y_T)      = sigma_Y^2 * T
        #   Cov(X_T,Y_T)  = rho * sigma_X * sigma_Y * (1 - e^{-kappa T}) / kappa
        T = np.asarray(T, dtype=float)
        kappa, sX, sY, rho = self.kappa, self.sigma_X, self.sigma_Y, self.rho
        var_X = (sX ** 2) / (2.0 * kappa) * (1.0 - np.exp(-2.0 * kappa * T))
        var_Y = (sY ** 2) * T
        cov_XY = rho * sX * sY * (1.0 - np.exp(-kappa * T)) / kappa
        return var_X + var_Y + 2.0 * cov_XY

    # ------------------------------------------------------------------
    # Jump contribution to log-forward
    # ------------------------------------------------------------------
    def _jump_log_contribution(self, T):
        """log G_jump(0, T) = lambda * integral_0^T [phi_J(e^{-kappa u}) - 1] du.

        Derivation sketch:
          X_T^jump = sum over jumps in (0,T] of e^{-kappa (T - tau_i)} * J_i
          E[exp(X_T^jump)] = MGF of a compound Poisson sum
                           = exp( lambda * integral_0^T (phi_J(e^{-kappa(T-s)}) - 1) ds )
          Substituting u = T - s gives the form above (integral over decay age u).

        We compute each T separately with scipy.integrate.quad because the
        integrand has no nice closed antiderivative when kappa > 0.
        """
        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        lam = self.jumps.intensity
        if lam == 0.0:
            out = np.zeros_like(T_arr)
        else:
            muJ = self.jumps.mean_log
            sJ = self.jumps.std_log
            kappa = self.kappa

            def integrand(u):
                # a(u) is how strongly a jump that is 'u' years old still affects X.
                # u=0 (just happened) -> a=1 -> full jump. u large -> a -> 0 -> faded out.
                a = np.exp(-kappa * u)
                # MGF of a Gaussian jump scaled by a: phi_J(a) = exp(a*mu_J + 0.5*a^2*sigma_J^2)
                return np.exp(a * muJ + 0.5 * (a ** 2) * (sJ ** 2)) - 1.0

            out = np.empty_like(T_arr)
            for i, Ti in enumerate(T_arr):
                # quad returns (value, abs_err); we keep only the value.
                val, _ = quad(integrand, 0.0, float(Ti))
                out[i] = lam * val

        # Preserve scalar-in -> scalar-out convention used elsewhere.
        return out.reshape(np.asarray(T).shape) if np.ndim(T) else out.item()

    # ------------------------------------------------------------------
    # Forward price — the centrepiece
    # ------------------------------------------------------------------
    def forward_price(self, T):
        """F(0, T) = E^Q[S_T]  — closed-form forward price.

        Decomposition (useful for teaching):
            log F(0,T) = f(T)                          # seasonality
                       + E[diff part]                  # mean of Gaussian piece
                       + 0.5 * Var[diff part]          # convexity for Gaussian piece
                       + log G_jump(T)                 # Merton convexity for jumps

        Shapes
        ------
        T : scalar OR 1-D array of delivery times, years.
        Returns float if T is scalar, else ndarray matching T's shape.
        """
        T_arr = np.asarray(T, dtype=float)
        seasonal = self.seasonality(T_arr)                 # shape: T_arr
        mu = self._diffusion_mean(T_arr)                   # shape: T_arr
        var = self._diffusion_var(T_arr)                   # shape: T_arr
        jump_log = np.asarray(self._jump_log_contribution(T_arr))
        log_F = seasonal + mu + 0.5 * var + jump_log
        F = np.exp(log_F)
        return float(F) if np.ndim(T) == 0 else F

    def forward_curve(self, T_array):
        """Vector-friendly alias: returns an ndarray of F(0, T) across T_array."""
        T_array = np.asarray(T_array, dtype=float)
        return self.forward_price(T_array)

    # ------------------------------------------------------------------
    # Diagnostic helper (useful for teaching / debugging)
    # ------------------------------------------------------------------
    def log_forward_components(self, T):
        """Break log F(0,T) into its four additive pieces. Returns a dict.

        Useful for seeing WHICH part of the model is driving the forward price
        (e.g., is that upswing seasonality, or jump-convexity?).
        """
        T_arr = np.asarray(T, dtype=float)
        return {
            "seasonality": self.seasonality(T_arr),
            "diffusion_mean": self._diffusion_mean(T_arr),
            "diffusion_convexity": 0.5 * self._diffusion_var(T_arr),
            "jump_convexity": np.asarray(self._jump_log_contribution(T_arr)),
        }
