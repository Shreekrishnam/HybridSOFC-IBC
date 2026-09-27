"""
turbomachinery.py
=================
Compressor and turbine isentropic models with thermodynamic property updates.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
from ..thermo import FluidProperties


def solve_compressor(
    P_in: float,
    H_in: float,
    P_out: float,
    eta_is: float,
    fluid: FluidProperties
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Solves an adiabatic compressor with isentropic efficiency eta_is.

    Returns:
        (H_out [J/kg], T_out [K], specific_work [J/kg], out_props)
    """
    in_props = fluid.flash_ph(P_in, H_in)
    s_in = in_props["s"]

    is_props = fluid.flash_ps(P_out, s_in)
    h_is = is_props["h"]

    w_is = h_is - H_in
    w_actual = w_is / eta_is
    h_out = H_in + w_actual

    out_props = fluid.flash_ph(P_out, h_out)
    return h_out, out_props["T"], w_actual, out_props


def solve_turbine(
    P_in: float,
    H_in: float,
    P_out: float,
    eta_is: float,
    fluid: FluidProperties
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Solves an adiabatic turbine with isentropic efficiency eta_is.

    Returns:
        (H_out [J/kg], T_out [K], specific_work [J/kg], out_props)
    """
    in_props = fluid.flash_ph(P_in, H_in)
    s_in = in_props["s"]

    is_props = fluid.flash_ps(P_out, s_in)
    h_is = is_props["h"]

    w_is = H_in - h_is
    w_actual = w_is * eta_is
    h_out = H_in - w_actual

    out_props = fluid.flash_ph(P_out, h_out)
    return h_out, out_props["T"], w_actual, out_props
