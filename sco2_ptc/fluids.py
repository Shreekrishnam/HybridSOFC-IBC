"""
fluids.py
=========
Single source of truth for the working-fluid set used by the optimisation campaigns.

Previously `mixture_optimization.DEFAULT_MIXTURES` and `run_three_paradigms.BASE_FLUIDS`
each carried their own table, and they had already diverged (91/9 versus 90/10). They
also carried hard-coded critical pressures used to set P_L, while the summary CSVs
reported the critical pressure REFPROP actually computes -- so the pressure a case ran
at and the pressure it was reported at could disagree. P_L is now derived from the
fluid's own critical point at run time.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence

from .thermo import FluidProperties


@dataclass(frozen=True)
class WorkingFluid:
    """A named composition. Critical properties come from REFPROP, never a literal."""
    name: str
    components: List[str]
    mole_fractions: List[float]

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier used for per-fluid logs and archives."""
        return (
            self.name.lower()
            .replace(" ", "_")
            .replace("%", "pct")
            .replace("+", "")
        )


# The CO2 + ethane ladder studied across all three comparison paradigms.
CAMPAIGN_FLUIDS: List[WorkingFluid] = [
    WorkingFluid("Pure sCO2", ["CO2"], [1.00]),
    WorkingFluid("95% CO2 + 5% C2H6", ["CO2", "ETHANE"], [0.95, 0.05]),
    WorkingFluid("90% CO2 + 10% C2H6", ["CO2", "ETHANE"], [0.90, 0.10]),
    WorkingFluid("85% CO2 + 15% C2H6", ["CO2", "ETHANE"], [0.85, 0.15]),
    WorkingFluid("80% CO2 + 20% C2H6", ["CO2", "ETHANE"], [0.80, 0.20]),
    WorkingFluid("75% CO2 + 25% C2H6", ["CO2", "ETHANE"], [0.75, 0.25]),
]

# Reference operating point that defines the "drop-in" paradigm.
P_L_BASELINE = 83.8e5
P_H_BASELINE = 220.0e5
PI_BASELINE = P_H_BASELINE / P_L_BASELINE  # 2.6253


def critical_pressure(fluid_spec: WorkingFluid) -> float:
    """Critical pressure [Pa] from REFPROP for this composition."""
    return FluidProperties(fluid_spec.components, fluid_spec.mole_fractions).P_crit


def paradigm_pressures(
    paradigm_id: int, P_crit: float, delta_P_L: float = 10.0e5
) -> tuple[float, float]:
    """(P_L, P_H) in Pa for a comparison paradigm, given the fluid's own P_crit.

    1. Fixed pressures      -- drop-in into existing turbomachinery.
    2. Critical proximity   -- P_L tracks the mixture, P_H pinned at the baseline.
    3. Normalised ratio     -- P_L tracks the mixture, pressure ratio pinned.

    Note for paradigms 2 and 3: for a *mixture* the single-phase margin is set by the
    cricondenbar, not the critical pressure, and for CO2 + ethane the phase envelope's
    maximum pressure exceeds P_crit. "P_crit + 10 bar" is therefore a weaker guarantee
    for a mixture than it is for pure CO2, and should be checked against the computed
    envelope before the margin is claimed in print.
    """
    if paradigm_id == 1:
        return P_L_BASELINE, P_H_BASELINE
    if paradigm_id == 2:
        return P_crit + delta_P_L, P_H_BASELINE
    if paradigm_id == 3:
        P_L = P_crit + delta_P_L
        return P_L, PI_BASELINE * P_L
    raise ValueError(f"Unknown paradigm_id: {paradigm_id}")
