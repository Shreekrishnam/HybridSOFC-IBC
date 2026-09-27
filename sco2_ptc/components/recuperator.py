"""
recuperator.py
==============
Discretized 1D counterflow recuperator finite-volume model.
Computes cell enthalpy residuals dH/dtau, heat duties Q_i, and local pressure drop chains.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

from ..thermo import FluidProperties
from ..correlations import cell_film_coefficient, cell_conductance


@dataclass
class RecuperatorState:
    """Complete profile across a discretized recuperator."""
    H_cold: np.ndarray      # (N,) cell enthalpies [J/kg]
    H_hot: np.ndarray       # (N,) cell enthalpies [J/kg]
    T_cold: np.ndarray      # (N,) cell temperatures [K]
    T_hot: np.ndarray       # (N,) cell temperatures [K]
    P_cold: np.ndarray      # (N,) cell pressures [Pa]
    P_hot: np.ndarray       # (N,) cell pressures [Pa]
    rho_cold: np.ndarray    # (N,) densities [kg/m^3]
    rho_hot: np.ndarray     # (N,) densities [kg/m^3]
    U_cell: np.ndarray      # (N,) overall heat transfer coeffs [W/(m^2 K)]
    Q_cell: np.ndarray      # (N,) heat transferred per cell [W]
    dT_cell: np.ndarray     # (N,) local temperature differences (T_hot - T_cold) [K]
    dP_cold: float          # Total cold side pressure drop [Pa]
    dP_hot: float           # Total hot side pressure drop [Pa]
    P_cold_out: float       # Cold outlet pressure [Pa]
    P_hot_out: float        # Hot outlet pressure [Pa]
    H_cold_out: float       # Cold outlet enthalpy [J/kg]
    H_hot_out: float        # Hot outlet enthalpy [J/kg]
    Q_total: float          # Total duty [W]
    dT_min: float           # Minimum pinch temperature difference [K]
    dT_avg: float           # Average pinch temperature difference [K]


class RecuperatorBlock:
    """Discretized counterflow recuperator block."""

    def __init__(
        self,
        name: str,
        N: int,
        L: float,
        D: float,
        A_cell: float,
        A_flow_tube: float,
        V_cell: float,
        R_cond: float,
        m_dot_tube_cold: float,
        m_dot_tube_hot: float,
        m_dot_cold: float,
        m_dot_hot: float,
        cell_offset: int = 0,
        use_muscl: bool = True,
        muscl_limiter: str = "van_leer",
    ):
        self.name = name
        self.N = N
        self.L = L
        self.D = D
        self.dL = L / N
        self.A_cell = A_cell
        self.A_flow_tube = A_flow_tube
        self.V_cell = V_cell
        self.R_cond = R_cond

        self.m_dot_tube_cold = m_dot_tube_cold
        self.m_dot_tube_hot = m_dot_tube_hot
        self.m_dot_cold = m_dot_cold
        self.m_dot_hot = m_dot_hot
        self.cell_offset = cell_offset
        self.use_muscl = use_muscl
        self.muscl_limiter = muscl_limiter

        # Cached denominators (frozen at initial values)
        self.inv_cap_cold: np.ndarray | None = None
        self.inv_cap_hot: np.ndarray | None = None

        # Cached U array for Jacobian evaluations
        self.cached_U: np.ndarray | None = None

        # Cached pressure profiles for Jacobian evaluations. Freezing these during
        # finite-difference columns does three things: perturbing one cell's enthalpy
        # no longer shifts every downstream cell pressure (so the per-cell property
        # cache hits), the declared Jacobian sparsity no longer needs the dense
        # upstream pressure chain, and dP/dh is genuinely negligible next to dQ/dh.
        self.cached_P_c: np.ndarray | None = None
        self.cached_P_h: np.ndarray | None = None
        self.cached_dP_cold: float = 0.0
        self.cached_dP_hot: float = 0.0
        self.cached_P_c_out: float = 0.0
        self.cached_P_h_out: float = 0.0

    def _slope_limiter(self, a: float, b: float) -> float:
        """Evaluates TVD slope limiter on adjacent differences a, b."""
        if a * b <= 0.0:
            return 0.0
        if self.muscl_limiter in ("van_leer", "muscl"):
            return 2.0 * a * b / (a + b)
        elif self.muscl_limiter == "minmod":
            return np.sign(a) * min(abs(a), abs(b))
        elif self.muscl_limiter == "van_albada":
            eps = 1e-4
            return ((a**2 + eps) * b + (b**2 + eps) * a) / (a**2 + b**2 + 2.0 * eps)
        return 0.0

    def init_capacities(self, rho_cold_init: np.ndarray, rho_hot_init: np.ndarray) -> None:
        """Initializes frozen denominators for pseudo-transient continuation."""
        self.inv_cap_cold = 1.0 / (rho_cold_init * self.V_cell)
        self.inv_cap_hot = 1.0 / (rho_hot_init * self.V_cell)

    def compute_residuals(
        self,
        H_c_cells: np.ndarray,
        H_h_cells: np.ndarray,
        H_c_in: float,
        P_c_in: float,
        H_h_in: float,
        P_h_in: float,
        fluid: FluidProperties,
        freeze_U: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, RecuperatorState]:
        """
        Evaluates dH_cold/dtau and dH_hot/dtau and returns complete state.
        Cold stream flows 0 -> N-1.
        Hot stream flows N-1 -> 0.
        """
        N = self.N
        dL = self.dL
        D = self.D
        A_flow = self.A_flow_tube

        # ---------------------------------------------------------------------
        # 1. March pressures along flow direction using local states
        # ---------------------------------------------------------------------
        # Cold side: flows 0 -> N-1 from P_c_in
        P_c = np.empty(N)
        T_c = np.empty(N)
        rho_c = np.empty(N)
        cp_c = np.empty(N)
        k_c = np.empty(N)
        mu_c = np.empty(N)
        dP_seg_c = np.empty(N)

        freeze_P = freeze_U and self.cached_P_c is not None

        if freeze_P:
            # Frozen profile: pressures do not respond to the enthalpy perturbation,
            # so unperturbed cells hit the exact-match per-cell property cache.
            P_c[:] = self.cached_P_c
            for i in range(N):
                props = fluid.get_cell_props(self.cell_offset + i, P_c[i], H_c_cells[i])
                T_c[i] = props["T"]
                rho_c[i] = props["rho"]
                cp_c[i] = props["cp"]
                k_c[i] = props["k"]
                mu_c[i] = props["mu"]
            P_c_out = self.cached_P_c_out
            dP_cold_tot = self.cached_dP_cold
        else:
            curr_P_c = P_c_in
            for i in range(N):
                c_idx = self.cell_offset + i
                props = fluid.get_cell_props(c_idx, curr_P_c, H_c_cells[i])
                P_c[i] = curr_P_c
                T_c[i] = props["T"]
                rho_c[i] = props["rho"]
                cp_c[i] = props["cp"]
                k_c[i] = props["k"]
                mu_c[i] = props["mu"]

                # Local friction and pressure drop for cell segment
                Re_c = self.m_dot_tube_cold * D / (A_flow * mu_c[i])
                Re_clamped = max(Re_c, 2300.0)
                f_c = (1.82 * np.log10(Re_clamped) - 1.64) ** (-2.0)
                dP_i = f_c * (dL / D) * (self.m_dot_tube_cold**2 / (2.0 * rho_c[i] * A_flow**2))
                dP_seg_c[i] = dP_i
                curr_P_c -= dP_i

            P_c_out = curr_P_c
            dP_cold_tot = P_c_in - P_c_out
            self.cached_P_c = P_c.copy()
            self.cached_P_c_out = P_c_out
            self.cached_dP_cold = dP_cold_tot

        # Hot side: flows N-1 -> 0 from P_h_in
        P_h = np.empty(N)
        T_h = np.empty(N)
        rho_h = np.empty(N)
        cp_h = np.empty(N)
        k_h = np.empty(N)
        mu_h = np.empty(N)
        dP_seg_h = np.empty(N)

        if freeze_P and self.cached_P_h is not None:
            P_h[:] = self.cached_P_h
            for i in range(N):
                props = fluid.get_cell_props(self.cell_offset + N + i, P_h[i], H_h_cells[i])
                T_h[i] = props["T"]
                rho_h[i] = props["rho"]
                cp_h[i] = props["cp"]
                k_h[i] = props["k"]
                mu_h[i] = props["mu"]
            P_h_out = self.cached_P_h_out
            dP_hot_tot = self.cached_dP_hot
        else:
            curr_P_h = P_h_in
            for i in range(N - 1, -1, -1):
                h_idx = self.cell_offset + N + i
                props = fluid.get_cell_props(h_idx, curr_P_h, H_h_cells[i])
                P_h[i] = curr_P_h
                T_h[i] = props["T"]
                rho_h[i] = props["rho"]
                cp_h[i] = props["cp"]
                k_h[i] = props["k"]
                mu_h[i] = props["mu"]

                Re_h = self.m_dot_tube_hot * D / (A_flow * mu_h[i])
                Re_clamped = max(Re_h, 2300.0)
                f_h = (1.82 * np.log10(Re_clamped) - 1.64) ** (-2.0)
                dP_i = f_h * (dL / D) * (self.m_dot_tube_hot**2 / (2.0 * rho_h[i] * A_flow**2))
                dP_seg_h[i] = dP_i
                curr_P_h -= dP_i

            P_h_out = curr_P_h
            dP_hot_tot = P_h_in - P_h_out
            self.cached_P_h = P_h.copy()
            self.cached_P_h_out = P_h_out
            self.cached_dP_hot = dP_hot_tot

        # ---------------------------------------------------------------------
        # 2. Overall conductance U per cell
        # ---------------------------------------------------------------------
        if freeze_U and self.cached_U is not None:
            U_cell = self.cached_U
        else:
            _, _, _, hc_c = cell_film_coefficient(
                self.m_dot_tube_cold, D, A_flow, mu_c, k_c, cp_c
            )
            _, _, _, hc_h = cell_film_coefficient(
                self.m_dot_tube_hot, D, A_flow, mu_h, k_h, cp_h
            )
            U_cell = cell_conductance(hc_c, hc_h, self.R_cond)
            self.cached_U = U_cell

        # ---------------------------------------------------------------------
        # 3. Cell Duties and Energy Residuals
        # ---------------------------------------------------------------------
        dT_cell = T_h - T_c
        Q_cell = U_cell * self.A_cell * dT_cell

        if not self.use_muscl:
            # Baseline 1st-order upwind differencing
            H_c_upwind = np.empty(N)
            H_c_upwind[0] = H_c_in
            H_c_upwind[1:] = H_c_cells[:-1]
            adv_cold = H_c_upwind - H_c_cells
            H_cold_out = H_c_cells[-1]

            H_h_upwind = np.empty(N)
            H_h_upwind[-1] = H_h_in
            H_h_upwind[:-1] = H_h_cells[1:]
            adv_hot = H_h_upwind - H_h_cells
            H_hot_out = H_h_cells[0]
        else:
            # 2nd-order MUSCL with TVD flux limiter
            # Cold stream (flows 0 -> N-1):
            slopes_c = np.zeros(N)
            for i in range(N):
                if i == 0:
                    a = 2.0 * (H_c_cells[0] - H_c_in)
                    b = H_c_cells[1] - H_c_cells[0] if N > 1 else a
                elif i == N - 1:
                    a = H_c_cells[N - 1] - H_c_cells[N - 2]
                    b = a
                else:
                    a = H_c_cells[i] - H_c_cells[i - 1]
                    b = H_c_cells[i + 1] - H_c_cells[i]
                slopes_c[i] = self._slope_limiter(a, b)

            H_c_faces = np.empty(N + 1)
            H_c_faces[0] = H_c_in
            for i in range(N - 1):
                H_c_faces[i + 1] = H_c_cells[i] + 0.5 * slopes_c[i]
            H_c_faces[N] = H_c_cells[-1] + 0.5 * slopes_c[-1]
            adv_cold = H_c_faces[:-1] - H_c_faces[1:]
            H_cold_out = H_c_faces[N]

            # Hot stream (flows N-1 -> 0):
            slopes_h = np.zeros(N)
            for i in range(N):
                if i == 0:
                    b = H_h_cells[1] - H_h_cells[0] if N > 1 else 0.0
                    a = b
                elif i == N - 1:
                    a = H_h_cells[N - 1] - H_h_cells[N - 2]
                    b = 2.0 * (H_h_in - H_h_cells[N - 1])
                else:
                    a = H_h_cells[i] - H_h_cells[i - 1]
                    b = H_h_cells[i + 1] - H_h_cells[i]
                slopes_h[i] = self._slope_limiter(a, b)

            H_h_faces = np.empty(N + 1)
            H_h_faces[0] = H_h_cells[0] - 0.5 * slopes_h[0]
            for i in range(N - 1):
                H_h_faces[i + 1] = H_h_cells[i + 1] - 0.5 * slopes_h[i + 1]
            H_h_faces[N] = H_h_in
            adv_hot = H_h_faces[1:] - H_h_faces[:-1]
            H_hot_out = H_h_faces[0]

        R_cold = self.m_dot_cold * adv_cold + Q_cell
        R_hot = self.m_dot_hot * adv_hot - Q_cell

        # Dynamic derivatives dH/dtau
        if self.inv_cap_cold is not None:
            dHc_dtau = R_cold * self.inv_cap_cold
            dHh_dtau = R_hot * self.inv_cap_hot
        else:
            dHc_dtau = R_cold / (rho_c * self.V_cell)
            dHh_dtau = R_hot / (rho_h * self.V_cell)

        # ---------------------------------------------------------------------
        # 4. Assemble State Profile
        # ---------------------------------------------------------------------
        state = RecuperatorState(
            H_cold=H_c_cells,
            H_hot=H_h_cells,
            T_cold=T_c,
            T_hot=T_h,
            P_cold=P_c,
            P_hot=P_h,
            rho_cold=rho_c,
            rho_hot=rho_h,
            U_cell=U_cell,
            Q_cell=Q_cell,
            dT_cell=dT_cell,
            dP_cold=dP_cold_tot,
            dP_hot=dP_hot_tot,
            P_cold_out=P_c_out,
            P_hot_out=P_h_out,
            H_cold_out=H_cold_out,
            H_hot_out=H_hot_out,
            Q_total=float(np.sum(Q_cell)),
            dT_min=float(np.min(dT_cell)),
            dT_avg=float(np.mean(dT_cell)),
        )

        return dHc_dtau, dHh_dtau, state


def compute_recuperator_residuals(
    recup: RecuperatorBlock,
    H_c_cells: np.ndarray,
    H_h_cells: np.ndarray,
    H_c_in: float,
    P_c_in: float,
    H_h_in: float,
    P_h_in: float,
    fluid: FluidProperties,
    freeze_U: bool = False
) -> Tuple[np.ndarray, np.ndarray, RecuperatorState]:
    """Helper functional wrapper for RecuperatorBlock."""
    return recup.compute_residuals(
        H_c_cells, H_h_cells, H_c_in, P_c_in, H_h_in, P_h_in, fluid, freeze_U=freeze_U
    )
