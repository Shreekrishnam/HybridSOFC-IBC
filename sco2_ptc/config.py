"""
config.py
=========
Configuration and parameter specifications for the sCO2 recompression cycle.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CycleConfig:
    """Fixed boundary conditions and cycle hardware parameters."""
    # Temperatures [K]
    T_cold: float = 300.0
    T_hot: float = 850.0

    # High pressure [Pa] (220 bar)
    P_H: float = 220.0e5

    # Low pressure offset above critical pressure [Pa] (10 bar)
    delta_P_L: float = 10.0e5

    # Optional explicit low pressure override [Pa]
    P_L_override: Optional[float] = None

    # Tube parameters
    N_tubes: int = 150
    D: float = 0.02  # Tube diameter [m]
    m_dot_tube_hot: float = 0.093  # Hot side tube mass flow [kg/s]

    # Isentropic efficiencies [-]
    eta_comp1: float = 0.88
    eta_comp2: float = 0.89
    eta_turb: float = 0.93

    # Stainless steel wall conduction [W/(m K)], [m]
    k_ss: float = 22.0
    L_ss: float = 0.008

    # Working fluid specification
    # Default mixture: 91% CO2 + 9% Ethane (mole fraction)
    fluid_components: list[str] = field(default_factory=lambda: ["CO2", "ETHANE"])
    x_CO2: float = 0.91  # Mole fraction of CO2

    # Molar masses [kg/kmol] or [g/mol]
    MW_CO2: float = 44.01
    MW_second: float = 30.07  # Ethane

    # C0.13 toggle: True evaluates state 1 at P_L (legacy EES behaviour),
    # False evaluates state 1 at P_8 = P_cooler_in (physically correct closure).
    use_legacy_p_in: bool = True

    # Spatial discretization advection scheme:
    # use_muscl: True enables 2nd-order MUSCL with flux limiter (default True)
    # Set to False to reproduce the 1st-order upwind model
    use_muscl: bool = True
    muscl_limiter: str = "van_leer"  # Options: 'van_leer', 'minmod', 'van_albada'

    def resolve_P_L(self, P_crit: float) -> float:
        """Low-side pressure [Pa]: the explicit override if given, else P_crit + delta_P_L."""
        return self.P_L_override if self.P_L_override is not None else (P_crit + self.delta_P_L)

    @property
    def R_cond(self) -> float:
        """Wall thermal resistance [m^2 K / W]. Counted once per cell."""
        return self.L_ss / self.k_ss

    @property
    def mole_fractions(self) -> list[float]:
        """Mole fractions of components."""
        if len(self.fluid_components) == 1:
            return [1.0]
        return [self.x_CO2, 1.0 - self.x_CO2]

    @property
    def mass_fractions(self) -> list[float]:
        """Mass fractions of components (z_CO2, 1 - z_CO2)."""
        if len(self.fluid_components) == 1:
            return [1.0]
        m1 = self.x_CO2 * self.MW_CO2
        m2 = (1.0 - self.x_CO2) * self.MW_second
        z_CO2 = m1 / (m1 + m2)
        return [z_CO2, 1.0 - z_CO2]


@dataclass
class CycleDOF:
    """Degrees of freedom for cycle optimization and simulation."""
    A_tube: float = 4.2      # Surface area per tube [m^2]
    r: float = 0.5           # Recuperator area fraction: A_LTR = r * A_tot [-]
    frac_recomp: float = 0.4 # Recompression fraction [-]
    N_cells: int = 80        # Discretization resolution per recuperator side


@dataclass
class CycleGeometry:
    """Computed geometric dimensions for recuperators and tubes."""
    A_tot: float
    A_LTR: float
    A_HTR: float
    A_flow_tube: float       # Single tube cross-sectional area [m^2]
    A_flow_total: float      # Total cross-sectional flow area [m^2]
    A_LTR_tube: float        # Area per tube in LTR [m^2]
    A_HTR_tube: float        # Area per tube in HTR [m^2]
    L_LTR: float             # Tube length in LTR [m]
    L_HTR: float             # Tube length in HTR [m]
    dL_LTR: float            # Cell length in LTR [m]
    dL_HTR: float            # Cell length in HTR [m]
    A_cell_LTR: float        # Heat exchange area per cell in LTR [m^2]
    A_cell_HTR: float        # Heat exchange area per cell in HTR [m^2]
    V_cell_LTR: float        # Fluid volume per cell (all tubes) in LTR [m^3]
    V_cell_HTR: float        # Fluid volume per cell (all tubes) in HTR [m^3]

    @classmethod
    def from_config_and_dof(cls, cfg: CycleConfig, dof: CycleDOF) -> CycleGeometry:
        A_tot = cfg.N_tubes * dof.A_tube
        A_LTR = dof.r * A_tot
        A_HTR = (1.0 - dof.r) * A_tot

        A_flow_tube = np.pi * cfg.D * cfg.D / 4.0
        A_flow_total = cfg.N_tubes * A_flow_tube

        A_LTR_tube = A_LTR / cfg.N_tubes
        A_HTR_tube = A_HTR / cfg.N_tubes

        L_LTR = A_LTR_tube / (np.pi * cfg.D)
        L_HTR = A_HTR_tube / (np.pi * cfg.D)

        dL_LTR = L_LTR / dof.N_cells
        dL_HTR = L_HTR / dof.N_cells

        A_cell_LTR = A_LTR / dof.N_cells
        A_cell_HTR = A_HTR / dof.N_cells

        V_cell_LTR = A_flow_total * dL_LTR
        V_cell_HTR = A_flow_total * dL_HTR

        return cls(
            A_tot=A_tot,
            A_LTR=A_LTR,
            A_HTR=A_HTR,
            A_flow_tube=A_flow_tube,
            A_flow_total=A_flow_total,
            A_LTR_tube=A_LTR_tube,
            A_HTR_tube=A_HTR_tube,
            L_LTR=L_LTR,
            L_HTR=L_HTR,
            dL_LTR=dL_LTR,
            dL_HTR=dL_HTR,
            A_cell_LTR=A_cell_LTR,
            A_cell_HTR=A_cell_HTR,
            V_cell_LTR=V_cell_LTR,
            V_cell_HTR=V_cell_HTR,
        )


@dataclass
class CycleFlows:
    """Mass flow rates across the cycle branches."""
    m_dot_tube_hot: float   # Hot stream mass flow per tube [kg/s]
    m_dot_tube_cold: float  # Cold stream mass flow per tube [kg/s]
    m_dot_hot: float        # Full circulating flow [kg/s]
    m_dot_cold: float       # Main branch through cooler and LTR cold side [kg/s]
    m_dot_recomp: float     # Recompression branch flow [kg/s]

    @classmethod
    def from_config_and_dof(cls, cfg: CycleConfig, dof: CycleDOF) -> CycleFlows:
        m_dot_tube_hot = cfg.m_dot_tube_hot
        m_dot_tube_cold = (1.0 - dof.frac_recomp) * m_dot_tube_hot
        m_dot_hot = cfg.N_tubes * m_dot_tube_hot
        m_dot_cold = cfg.N_tubes * m_dot_tube_cold
        m_dot_recomp = m_dot_hot - m_dot_cold

        return cls(
            m_dot_tube_hot=m_dot_tube_hot,
            m_dot_tube_cold=m_dot_tube_cold,
            m_dot_hot=m_dot_hot,
            m_dot_cold=m_dot_cold,
            m_dot_recomp=m_dot_recomp,
        )
