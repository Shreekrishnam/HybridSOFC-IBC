"""
exergy.py
=========
Post-convergence Second-Law (Exergy and Entropy Generation) Analysis
for the Supercritical CO2 Recompression Brayton Cycle.

NOTE: As per design, this analysis is strictly evaluated POST-CONVERGENCE
and is never invoked during intermediate pseudo-transient continuation steps.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from .thermo import FluidProperties
from .flowsheet import FlowsheetCycleState


@dataclass
class ExergyComponentResult:
    name: str
    S_dot_gen: float        # Entropy generation rate [W/K]
    E_dot_dest: float       # Exergy destruction rate [W]
    dest_fraction: float    # Percentage of total cycle exergy destruction [%]


@dataclass
class CycleExergyAnalysis:
    T0: float                                   # Dead state / ambient temperature [K]
    P0: float                                   # Dead state / ambient pressure [Pa]
    components: Dict[str, ExergyComponentResult] # Component-wise breakdown
    S_dot_gen_total: float                      # Total cycle entropy generation [W/K]
    E_dot_dest_total: float                     # Total cycle exergy destruction [W]
    eta_thermal: float                          # First-law thermal efficiency [-]
    eta_exergetic: float                        # Second-law exergetic efficiency [-]

    def summary(self) -> str:
        lines = [
            "=" * 76,
            f"  SECOND-LAW EXERGY & ENTROPY GENERATION ANALYSIS (T0 = {self.T0:.2f} K)",
            "=" * 76,
            f"  {'Component':<24} {'S_dot_gen (W/K)':>16} {'E_dot_dest (kW)':>16} {'Share (%)':>12}",
            "-" * 76,
        ]
        for name, c in self.components.items():
            lines.append(
                f"  {name:<24} {c.S_dot_gen:>16.2f} {c.E_dot_dest / 1e3:>16.2f} {c.dest_fraction:>11.2f} %"
            )
        lines.extend([
            "-" * 76,
            f"  {'TOTAL CYCLE DESTRUCTION':<24} {self.S_dot_gen_total:>16.2f} {self.E_dot_dest_total / 1e3:>16.2f} {100.0:>11.2f} %",
            "=" * 76,
            f"  First-Law Thermal Efficiency (eta_th)   : {self.eta_thermal * 100:.3f} %",
            f"  Second-Law Exergetic Efficiency (eta_ex): {self.eta_exergetic * 100:.3f} %",
            "=" * 76,
        ])
        return "\n".join(lines)


def compute_cycle_exergy(
    state: FlowsheetCycleState,
    fluid: FluidProperties,
    T0: float = 298.15,
    P0: float = 1.01325e5,
    T_source: Optional[float] = None,
    T_sink: Optional[float] = None,
) -> CycleExergyAnalysis:
    """
    Computes exact entropy generation rates (S_dot_gen) and exergy destruction (E_dot_dest = T0 * S_dot_gen)
    across all cycle components at the converged steady state.

    Parameters:
    -----------
    state : FlowsheetCycleState (converged state from PTC solver)
    fluid : FluidProperties
    T0 : float, reference dead-state temperature [K] (default 298.15 K)
    P0 : float, reference dead-state pressure [Pa] (default 1.01325 bar)
    T_source : float, heat source boundary temperature [K] (default T_hot + 30 K)
    T_sink : float, heat sink boundary temperature [K] (default T_cold - 10 K)
    """
    st = state.stations

    # Flash entropy for all 12 stations (post-convergence only)
    S = {}
    for idx in range(1, 13):
        props = fluid.flash_ph(st[idx]["P"], st[idx]["h"])
        S[idx] = props["s"]  # [J/(kg K)]

    m_hot = st[5]["m_dot"]       # Total mass flow [kg/s]
    m_cold = st[1]["m_dot"]      # Split cold mass flow [kg/s]
    m_recomp = st[3]["m_dot"]    # Recompression mass flow [kg/s]

    if T_source is None:
        T_source = st[5]["T"] + 30.0  # Conservative heat source temperature
    if T_sink is None:
        T_sink = min(st[1]["T"] - 10.0, T0)  # Cooling water/ambient heat sink temperature

    # -------------------------------------------------------------------------
    # 1. Component Entropy Generation Rates [W/K]
    # -------------------------------------------------------------------------
    # Turbine (adiabatic)
    S_gen_turb = max(0.0, m_hot * (S[6] - S[5]))

    # Main Compressor 1 (adiabatic)
    S_gen_comp1 = max(0.0, m_cold * (S[2] - S[1]))

    # Recompressor 2 (adiabatic)
    S_gen_comp2 = max(0.0, m_recomp * (S[3] - S[8]))

    # High Temperature Recuperator (HTR, adiabatic shell)
    # Cold: 12 -> 4; Hot: 6 -> 7
    S_gen_htr = max(0.0, m_hot * (S[4] - S[12]) + m_hot * (S[7] - S[6]))

    # Low Temperature Recuperator (LTR, adiabatic shell)
    # Cold: 2 -> 11; Hot: 7 -> 8
    S_gen_ltr = max(0.0, m_cold * (S[11] - S[2]) + m_hot * (S[8] - S[7]))

    # Mixing Junction (adiabatic)
    # Inlets: 11 (cold branch) + 3 (recomp branch); Outlet: 12
    S_gen_mixer = max(0.0, m_hot * S[12] - (m_cold * S[11] + m_recomp * S[3]))

    # Primary Heater: 4 -> 5 with heat addition Q_dot_in from T_source
    Q_in = state.Q_dot_in
    S_gen_heater = max(0.0, m_hot * (S[5] - S[4]) - (Q_in / T_source))

    # Pre-cooler: 8 -> 1 with heat rejection Q_dot_out to T_sink
    Q_out = state.Q_dot_out
    S_gen_cooler = max(0.0, m_cold * (S[1] - S[8]) + (Q_out / T_sink))

    raw_components = {
        "Turbine": S_gen_turb,
        "Main Compressor (MC)": S_gen_comp1,
        "Recompressor (RC)": S_gen_comp2,
        "HTR Recuperator": S_gen_htr,
        "LTR Recuperator": S_gen_ltr,
        "Mixing Junction": S_gen_mixer,
        "Primary Heater": S_gen_heater,
        "Pre-Cooler": S_gen_cooler,
    }

    S_dot_gen_total = sum(raw_components.values())
    E_dot_dest_total = T0 * S_dot_gen_total

    components = {}
    for name, s_gen in raw_components.items():
        e_dest = T0 * s_gen
        fraction = (e_dest / E_dot_dest_total * 100.0) if E_dot_dest_total > 0 else 0.0
        components[name] = ExergyComponentResult(
            name=name,
            S_dot_gen=s_gen,
            E_dot_dest=e_dest,
            dest_fraction=fraction,
        )

    # Exergy input from source
    # E_in = Q_in * (1 - T0 / T_source)
    E_in = Q_in * (1.0 - T0 / T_source) if T_source > T0 else Q_in
    W_net = state.W_dot_net
    eta_exergetic = (W_net / E_in) if E_in > 0 else 0.0

    return CycleExergyAnalysis(
        T0=T0,
        P0=P0,
        components=components,
        S_dot_gen_total=S_dot_gen_total,
        E_dot_dest_total=E_dot_dest_total,
        eta_thermal=state.eta,
        eta_exergetic=eta_exergetic,
    )
