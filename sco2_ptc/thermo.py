"""
thermo.py
=========
Thermodynamic property interface using ctREFPROP with caching.
All units are in SI mass base:
- Temperature: T [K]
- Pressure: P [Pa] (note: 1 bar = 1e5 Pa)
- Density: rho [kg/m^3]
- Enthalpy: h [J/kg]
- Entropy: s [J/(kg K)]
- Isobaric specific heat: cp [J/(kg K)]
- Thermal conductivity: k [W/(m K)]
- Dynamic viscosity: mu [Pa s]
- Prandtl number: Pr [-]
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Ensure DynamicHX path is available for refprop_fluid if installed there
_DYNHX_PATH = Path(r"D:\OneDrive - Indian Institute of Science\Research-Analysis\04 Models\DynamicHX\src")
if _DYNHX_PATH.exists() and str(_DYNHX_PATH) not in sys.path:
    sys.path.insert(0, str(_DYNHX_PATH))

try:
    from dynhx.properties.refprop_fluid import REFPROPFluid, is_undef
except ImportError:
    # Fallback directly to ctREFPROP if refprop_fluid not in path
    from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
    REFPROPFluid = None


# REFPROP component name -> CoolProp HEOS name.
# Used only for the near-critical (P, T) -> h bridge below. A component that is
# absent from this map disables the CoolProp path entirely rather than silently
# substituting a different fluid.
_COOLPROP_NAMES = {
    "CO2": "CarbonDioxide",
    "CARBONDIOXIDE": "CarbonDioxide",
    "ETHANE": "Ethane",
    "ARGON": "Argon",
    "SO2": "SulfurDioxide",
    "SULFURDIOXIDE": "SulfurDioxide",
    "NITROGEN": "Nitrogen",
    "N2": "Nitrogen",
    "OXYGEN": "Oxygen",
    "O2": "Oxygen",
    "KRYPTON": "Krypton",
    "XENON": "Xenon",
    "HELIUM": "Helium",
}


def coolprop_fluid_string(components: list[str]) -> "str | None":
    """Maps REFPROP component names to a CoolProp '&'-joined fluid string.

    Returns None if any component is unmapped, so callers can disable the
    CoolProp path instead of evaluating a different mixture.
    """
    names = []
    for c in components:
        cp_name = _COOLPROP_NAMES.get(str(c).strip().upper())
        if cp_name is None:
            return None
        names.append(cp_name)
    return "&".join(names)


class FluidProperties:
    """
    Fixed-composition thermodynamic property calculator backed by REFPROP 10.
    Provides fast caching for pseudo-transient continuation and finite-difference Jacobians.
    """

    # Validity window enforced by flash_ph. Inputs outside it are clamped, which
    # flattens the residual; n_clamped_P / n_clamped_h count how often that happens
    # so a solve that "converged" against a clamp can be recognised as such.
    P_MIN = 65.0e5
    P_MAX = 250.0e5
    H_MIN = 190.0e3
    H_MAX = 1350.0e3

    def __init__(self, components: list[str], mole_fractions: list[float]):
        self.components = list(components)
        self.mole_fractions = list(mole_fractions)
        self.fluid_str = " * ".join(components)

        if REFPROPFluid is not None:
            self._fluid = REFPROPFluid(components, mole_fractions)
            self._rp = self._fluid._rp
            self._si = self._fluid._si
            self._z20 = self._fluid._z20
        else:
            rp_path = os.environ.get("REFPROP_PATH", r"C:\Program Files (x86)\REFPROP")
            self._rp = REFPROPFunctionLibrary(rp_path)
            self._rp.SETPATHdll(rp_path)
            self._si = self._rp.GETENUMdll(0, "MASS BASE SI").iEnum
            self._z20 = list(mole_fractions) + [0.0] * (20 - len(mole_fractions))
            self._fluid = None

        # Calculate critical properties
        r_crit = self._rp.REFPROPdll(self.fluid_str, "CRIT", "T;P;D", self._si, 0, 0, 0, 0, self._z20)
        self.T_crit = float(r_crit.Output[0])
        self.P_crit = float(r_crit.Output[1])  # Pa
        self.rho_crit = float(r_crit.Output[2])

        # Cell-level cache for sparse finite-difference Jacobian speedup
        self._cell_cache_P: dict[int, float] = {}
        self._cell_cache_H: dict[int, float] = {}
        self._cell_cache_res: dict[int, Dict[str, float]] = {}

        # Global lookup cache
        self._cache_ph: dict[Tuple[float, float], Dict[str, float]] = {}
        self._cache_pt: dict[Tuple[float, float], Dict[str, float]] = {}

        # Initialize CoolProp HEOS state for robust near-critical PT conversions.
        # Disabled (None) whenever any component has no CoolProp equivalent, so an
        # unmapped fluid falls back to REFPROP rather than silently evaluating a
        # different mixture.
        self._cp_fluid_str = coolprop_fluid_string(self.components)
        self._cp_state = None
        if self._cp_fluid_str is not None:
            try:
                import CoolProp.CoolProp as CP
                self._cp_state = CP.AbstractState("HEOS", self._cp_fluid_str)
                if len(self.mole_fractions) > 1:
                    self._cp_state.set_mole_fractions(self.mole_fractions)
            except Exception:
                self._cp_state = None

        # Diagnostics: number of flash_ph calls whose (P, h) inputs hit a clamp.
        self.n_clamped_P = 0
        self.n_clamped_h = 0

    def get_critical_props(self) -> Tuple[float, float, float]:
        """Returns (T_crit [K], P_crit [Pa], rho_crit [kg/m^3])."""
        return self.T_crit, self.P_crit, self.rho_crit

    def clear_cell_cache(self) -> None:
        """Clears the per-cell perturbation cache."""
        self._cell_cache_P.clear()
        self._cell_cache_H.clear()
        self._cell_cache_res.clear()

    def get_cell_props(self, cell_idx: int, P_pa: float, H_jkg: float) -> Dict[str, float]:
        """
        Fast cell-level property fetch.
        If cell_idx has identical (P, H) as its last call, returns cached result instantly.
        Used extensively by Jacobian finite differences.
        """
        if (cell_idx in self._cell_cache_P and
            self._cell_cache_P[cell_idx] == P_pa and
            self._cell_cache_H[cell_idx] == H_jkg):
            return self._cell_cache_res[cell_idx]

        res = self.flash_ph(P_pa, H_jkg)
        self._cell_cache_P[cell_idx] = P_pa
        self._cell_cache_H[cell_idx] = H_jkg
        self._cell_cache_res[cell_idx] = res
        return res

    def flash_ph(self, P_pa: float, h_jkg: float) -> Dict[str, float]:
        """
        PH flash: evaluates temperature, density, and transport properties.
        Clamps inputs to valid single-phase supercritical EOS range to prevent
        solver trial steps from venturing into cryogenic/unphysical domains.
        """
        P_in = float(P_pa)
        h_in = float(h_jkg)
        P_safe = min(max(P_in, self.P_MIN), self.P_MAX)
        h_safe = min(max(h_in, self.H_MIN), self.H_MAX)
        if P_safe != P_in:
            self.n_clamped_P += 1
        if h_safe != h_in:
            self.n_clamped_h += 1

        key = (P_safe, h_safe)
        if key in self._cache_ph:
            return self._cache_ph[key]

        try:
            if self._fluid is not None:
                res = self._fluid.update_PH(P_safe, h_safe)
                d = {
                    "T": res["T"],
                    "rho": res["rho"],
                    "h": res["h"],
                    "s": res["s"],
                    "cp": res["cp"],
                    "cv": res["cv"],
                    "mu": res["eta"],
                    "k": res["kap"],
                    "Pr": res["Pr"],
                    "phase": res["phase"],
                }
            else:
                r = self._rp.REFPROPdll(
                    self.fluid_str, "PH", "T;D;H;S;CP;CV;VIS;TCX;PRANDTL",
                    self._si, 0, 0, P_safe, h_safe, self._z20
                )
                d = {
                    "T": float(r.Output[0]),
                    "rho": float(r.Output[1]),
                    "h": float(r.Output[2]),
                    "s": float(r.Output[3]),
                    "cp": float(r.Output[4]),
                    "cv": float(r.Output[5]),
                    "mu": float(r.Output[6]),
                    "k": float(r.Output[7]),
                    "Pr": float(r.Output[8]),
                    "phase": "single-phase" if r.ierr <= 0 else "two-phase",
                }
        except RuntimeError:
            # If evaluation fails near boundary, fallback to safer mid-range state
            h_fallback = min(max(h_safe, 250000.0), 1200000.0)
            res = self._fluid.update_PH(P_safe, h_fallback) if self._fluid else None
            if res is not None:
                d = {
                    "T": res["T"], "rho": res["rho"], "h": res["h"], "s": res["s"],
                    "cp": res["cp"], "cv": res["cv"], "mu": res["eta"], "k": res["kap"],
                    "Pr": res["Pr"], "phase": res["phase"],
                }
            else:
                raise

        self._cache_ph[key] = d
        return d

    # Temperature below which flash_pt prefers the CoolProp -> PH bridge over
    # REFPROP's TPFL2 solver. Above it TPFL2 is well behaved and cheaper.
    T_PT_BRIDGE = 350.0

    def _flash_pt_via_ph(self, P_pa: float, T_k: float) -> Optional[Dict[str, float]]:
        """Solves (P, T) by bridging through CoolProp for an h estimate, then
        Newton-iterating REFPROP's PH flash until the returned T matches T_k.

        A single Newton step is not enough across the pseudo-critical ridge, where
        cp varies steeply: the corrected state can still sit a few tenths of a K
        off target, and this state sets the main-compressor inlet. Returns None if
        the bridge is unavailable or fails to converge, so the caller can fall back
        to REFPROP rather than accept an off-target state.
        """
        if self._cp_state is None:
            return None
        try:
            import CoolProp.CoolProp as CP
            self._cp_state.update(CP.PT_INPUTS, P_pa, T_k)
            h = float(self._cp_state.hmass())
        except Exception:
            return None

        try:
            res = self.flash_ph(P_pa, h)
            for _ in range(self.PT_BRIDGE_MAX_ITER):
                dT = res["T"] - T_k
                if abs(dT) <= self.PT_BRIDGE_TOL:
                    return res
                cp = res["cp"]
                if not np.isfinite(cp) or cp <= 0.0:
                    return None
                h = res["h"] - cp * dT
                res = self.flash_ph(P_pa, h)
            # Did not reach tolerance within the iteration budget.
            return res if abs(res["T"] - T_k) <= self.PT_BRIDGE_TOL else None
        except Exception:
            return None

    PT_BRIDGE_TOL = 1e-6      # [K] agreement required between returned T and T_k
    PT_BRIDGE_MAX_ITER = 8

    def flash_pt(self, P_pa: float, T_k: float) -> Dict[str, float]:
        """
        PT flash: evaluates properties given pressure and temperature.
        """
        key = (P_pa, T_k)
        if key in self._cache_pt:
            return self._cache_pt[key]

        # For near-critical/retrograde temperatures, avoid REFPROP's TPFL2 solver
        # by converting (P, T) -> h via CoolProp HEOS and evaluating REFPROP's PH flash.
        if T_k < self.T_PT_BRIDGE and self._cp_state is not None:
            res = self._flash_pt_via_ph(P_pa, T_k)
            if res is not None:
                self._cache_pt[key] = res
                return res

        if self._fluid is not None:
            try:
                res = self._fluid.update_PT(P_pa, T_k)
                d = {
                    "T": res["T"],
                    "rho": res["rho"],
                    "h": res["h"],
                    "s": res["s"],
                    "cp": res["cp"],
                    "cv": res["cv"],
                    "mu": res["eta"],
                    "k": res["kap"],
                    "Pr": res["Pr"],
                    "phase": res["phase"],
                }
            except RuntimeError:
                # If REFPROP's TPFL2 solver fails near the critical point (e.g. 90% CO2 at
                # 300 K), bridge through CoolProp HEOS to get h at (P, T), then use REFPROP's
                # PH flash, which is far better behaved for dense supercritical single phase.
                d = self._flash_pt_via_ph(P_pa, T_k)
                if d is None:
                    raise
        else:
            r = self._rp.REFPROPdll(
                self.fluid_str, "TP", "D;H;S;CP;CV;VIS;TCX;PRANDTL",
                self._si, 0, 0, T_k, P_pa, self._z20
            )
            d = {
                "T": T_k,
                "rho": float(r.Output[0]),
                "h": float(r.Output[1]),
                "s": float(r.Output[2]),
                "cp": float(r.Output[3]),
                "cv": float(r.Output[4]),
                "mu": float(r.Output[5]),
                "k": float(r.Output[6]),
                "Pr": float(r.Output[7]),
                "phase": "single-phase" if r.ierr <= 0 else "two-phase",
            }

        self._cache_pt[key] = d
        return d

    def flash_ps(self, P_pa: float, s_jkgk: float) -> Dict[str, float]:
        """
        PS flash: evaluates isentropic discharge state given pressure and entropy.
        """
        if self._fluid is not None:
            res = self._fluid.update_PS(P_pa, s_jkgk)
            return {
                "T": res["T"],
                "rho": res["rho"],
                "h": res["h"],
                "s": res["s"],
                "cp": res["cp"],
                "cv": res["cv"],
                "mu": res["eta"],
                "k": res["kap"],
                "Pr": res["Pr"],
                "phase": res["phase"],
            }
        else:
            r = self._rp.REFPROPdll(
                self.fluid_str, "PS", "T;D;H;S;CP;CV;VIS;TCX;PRANDTL",
                self._si, 0, 0, P_pa, s_jkgk, self._z20
            )
            return {
                "T": float(r.Output[0]),
                "rho": float(r.Output[1]),
                "h": float(r.Output[2]),
                "s": float(r.Output[3]),
                "cp": float(r.Output[4]),
                "cv": float(r.Output[5]),
                "mu": float(r.Output[6]),
                "k": float(r.Output[7]),
                "Pr": float(r.Output[8]),
                "phase": "single-phase" if r.ierr <= 0 else "two-phase",
            }
