"""
junctions.py
============
Flow junctions: mixer, flow split, heater, and pre-cooler models.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
from ..thermo import FluidProperties


def solve_mixer(
    H_cold_branch: float,
    m_dot_cold: float,
    H_recomp_branch: float,
    m_dot_recomp: float
) -> float:
    """
    Adiabatic mixing junction: computes mass-weighted mixed enthalpy.
        H_mix = (m_dot_cold * H_cold + m_dot_recomp * H_recomp) / (m_dot_cold + m_dot_recomp)
    """
    total_flow = m_dot_cold + m_dot_recomp
    if total_flow <= 0.0:
        raise ValueError(f"Total mixing flow must be positive, got {total_flow}")
    return (m_dot_cold * H_cold_branch + m_dot_recomp * H_recomp_branch) / total_flow


def solve_flow_split(
    H_in: float,
    frac_recomp: float
) -> Tuple[float, float]:
    """
    Isenthalpic flow split:
    Returns (H_recomp_branch, H_cooler_branch). Both equal H_in.
    """
    return H_in, H_in


def solve_heater(
    P_in: float,
    H_in: float,
    T_hot: float,
    m_dot_hot: float,
    fluid: FluidProperties
) -> Tuple[float, float, Dict[str, float]]:
    """
    Top cycle heater: heats the fluid to T_hot at constant pressure P_in.
    Returns:
        (H_out [J/kg], Q_dot_in [W], out_props)
    """
    out_props = fluid.flash_pt(P_in, T_hot)
    H_out = out_props["h"]
    Q_dot_in = m_dot_hot * (H_out - H_in)
    return H_out, Q_dot_in, out_props


def solve_cooler(
    P_in: float,
    H_in: float,
    T_cold: float,
    m_dot_cold: float,
    fluid: FluidProperties,
    P_eval: float | None = None
) -> Tuple[float, float, Dict[str, float]]:
    """
    Pre-cooler: cools the fluid to T_cold.
    If P_eval is provided, the cold state properties are evaluated at P_eval
    (e.g. legacy P_L for parity check vs C0.13), otherwise at P_in.
    Returns:
        (H_out [J/kg], Q_dot_out [W], out_props)
    """
    p_use = P_eval if P_eval is not None else P_in
    out_props = fluid.flash_pt(p_use, T_cold)
    H_out = out_props["h"]
    Q_dot_out = m_dot_cold * (H_in - H_out)
    return H_out, Q_dot_out, out_props
