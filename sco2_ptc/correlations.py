"""
correlations.py
===============
Vectorized transport correlations matching the sCO2 cycle equations:
- Petukhov/Filonenko friction factor
- Gnielinski Nusselt number
- Combined overall heat transfer coefficient (single wall resistance)
"""

from __future__ import annotations
import numpy as np


def fanning_friction_factor(Re: np.ndarray | float) -> np.ndarray | float:
    """
    Petukhov/Filonenko friction factor for turbulent pipe flow:
        f = (1.82 * log10(Re) - 1.64)^(-2)
    Valid for 2300 < Re < 5e6. Clamped to Re >= 2300.
    """
    Re_clamped = np.maximum(Re, 2300.0)
    return (1.82 * np.log10(Re_clamped) - 1.64) ** (-2.0)


def gnielinski_nusselt(
    Re: np.ndarray | float,
    Pr: np.ndarray | float,
    f: np.ndarray | float | None = None
) -> np.ndarray | float:
    """
    Gnielinski correlation for turbulent heat transfer in circular tubes:
        Nu = ((f/8) * (Re - 1000) * Pr) / (1 + 12.7 * sqrt(f/8) * (Pr^(2/3) - 1))
    """
    if f is None:
        f = fanning_friction_factor(Re)

    Re_eff = np.maximum(Re, 2300.0)
    Pr_eff = np.maximum(Pr, 0.6)
    f_8 = f / 8.0
    sqrt_f_8 = np.sqrt(f_8)

    num = f_8 * (Re_eff - 1000.0) * Pr_eff
    den = 1.0 + 12.7 * sqrt_f_8 * (Pr_eff ** (2.0 / 3.0) - 1.0)
    Nu = num / den
    return np.maximum(Nu, 4.36)


def cell_film_coefficient(
    flow_per_tube: float,
    D: float,
    A_flow_tube: float,
    mu: np.ndarray | float,
    k: np.ndarray | float,
    cp: np.ndarray | float
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """
    Computes local Re, f, Nu, hc for a stream.
    Returns (Re, f, Nu, hc).
    """
    Re = flow_per_tube * D / (A_flow_tube * mu)
    f = fanning_friction_factor(Re)
    Pr = mu * cp / k
    Nu = gnielinski_nusselt(Re, Pr, f)
    hc = Nu * k / D
    return Re, f, Nu, hc


def cell_conductance(
    hc_cold: np.ndarray | float,
    hc_hot: np.ndarray | float,
    R_cond: float
) -> np.ndarray | float:
    """
    Overall heat transfer coefficient U [W/(m^2 K)] with single wall resistance:
        1 / U = 1 / hc_cold + 1 / hc_hot + R_cond
    """
    return 1.0 / (1.0 / hc_cold + 1.0 / hc_hot + R_cond)
