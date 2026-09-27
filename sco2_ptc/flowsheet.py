"""
flowsheet.py
============
Master flowsheet assembling the full sCO2 recompression Brayton cycle into
a single pseudo-transient continuation (PTC) system of ODEs.

State vector:
    Y = [ H_HTR_c(0..N-1) | H_HTR_h(0..N-1) | H_LTR_c(0..N-1) | H_LTR_h(0..N-1) | H_mix ]
Dimension: 4*N + 1
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix

from .config import CycleConfig, CycleDOF, CycleGeometry, CycleFlows
from .thermo import FluidProperties
from .components.turbomachinery import solve_compressor, solve_turbine
from .components.junctions import solve_mixer, solve_heater, solve_cooler
from .components.recuperator import RecuperatorBlock, RecuperatorState


@dataclass
class FlowsheetCycleState:
    """Detailed cycle state containing all components and thermodynamic stations."""
    # Stations 1..12 properties
    stations: Dict[int, Dict[str, float]]
    # Component duty / power metrics
    W_dot_turb: float
    W_dot_comp_1: float
    W_dot_comp_2: float
    W_dot_net: float
    Q_dot_in: float
    Q_dot_out: float
    eta: float
    # Energy balance error
    energy_balance_err: float
    # Recuperator states
    ltr_state: RecuperatorState
    htr_state: RecuperatorState


class Flowsheet:
    """
    Manages the cycle simulation, state layout, Jacobian sparsity,
    and the master ODE RHS evaluation.
    """

    def __init__(self, cfg: CycleConfig, dof: CycleDOF, fluid: FluidProperties):
        self.cfg = cfg
        self.dof = dof
        self.fluid = fluid

        self.geom = CycleGeometry.from_config_and_dof(cfg, dof)
        self.flows = CycleFlows.from_config_and_dof(cfg, dof)

        N = dof.N_cells
        self.N = N
        self.n_states = 4 * N + 1

        # Slices in master state vector Y
        self.sl_htr_c = slice(0, N)
        self.sl_htr_h = slice(N, 2 * N)
        self.sl_ltr_c = slice(2 * N, 3 * N)
        self.sl_ltr_h = slice(3 * N, 4 * N)
        self.idx_mix = 4 * N

        # Mixing junction relaxation time constant [s]
        self.tau_mix = 0.5

        # Instantiate recuperator blocks
        # cell_offsets ensure unique cache keys in FluidProperties
        self.htr_block = RecuperatorBlock(
            name="HTR",
            N=N,
            L=self.geom.L_HTR,
            D=cfg.D,
            A_cell=self.geom.A_cell_HTR,
            A_flow_tube=self.geom.A_flow_tube,
            V_cell=self.geom.V_cell_HTR,
            R_cond=cfg.R_cond,
            m_dot_tube_cold=self.flows.m_dot_tube_hot,
            m_dot_tube_hot=self.flows.m_dot_tube_hot,
            m_dot_cold=self.flows.m_dot_hot,
            m_dot_hot=self.flows.m_dot_hot,
            cell_offset=0,
            use_muscl=cfg.use_muscl,
            muscl_limiter=cfg.muscl_limiter,
        )

        self.ltr_block = RecuperatorBlock(
            name="LTR",
            N=N,
            L=self.geom.L_LTR,
            D=cfg.D,
            A_cell=self.geom.A_cell_LTR,
            A_flow_tube=self.geom.A_flow_tube,
            V_cell=self.geom.V_cell_LTR,
            R_cond=cfg.R_cond,
            m_dot_tube_cold=self.flows.m_dot_tube_cold,
            m_dot_tube_hot=self.flows.m_dot_tube_hot,
            m_dot_cold=self.flows.m_dot_cold,
            m_dot_hot=self.flows.m_dot_hot,
            cell_offset=2 * N,
            use_muscl=cfg.use_muscl,
            muscl_limiter=cfg.muscl_limiter,
        )

        # Resolve and validate the low-side pressure once. flash_ph silently clamps
        # its inputs to [P_MIN, P_MAX], which flattens the residual and lets a solve
        # "converge" against the clamp, so a P outside that window is refused here
        # rather than discovered as a wrong answer later.
        self.P_L = cfg.resolve_P_L(fluid.P_crit)
        for label, value in (("P_L", self.P_L), ("P_H", cfg.P_H)):
            if not (FluidProperties.P_MIN <= value <= FluidProperties.P_MAX):
                raise ValueError(
                    f"{label} = {value / 1e5:.2f} bar is outside the property window "
                    f"[{FluidProperties.P_MIN / 1e5:.1f}, {FluidProperties.P_MAX / 1e5:.1f}] bar "
                    f"enforced by FluidProperties.flash_ph; results would be clamped."
                )
        if self.P_L >= cfg.P_H:
            raise ValueError(
                f"P_L = {self.P_L / 1e5:.2f} bar must be below P_H = {cfg.P_H / 1e5:.2f} bar."
            )

        # Build Jacobian sparsity
        self.jac_sparsity = self.build_jac_sparsity()

        # Last evaluated full state
        self.last_cycle_state: Optional[FlowsheetCycleState] = None

    def build_initial_guess(self) -> np.ndarray:
        """
        Builds an initial state vector Y0 from baseline thermodynamic estimates.
        Linear interpolation between inlet and expected outlet temperatures.
        """
        N = self.N
        cfg = self.cfg
        fluid = self.fluid

        # Flash key reference points
        P_L = self.P_L
        P_H = cfg.P_H

        # State 1 estimate: T_cold, P_L
        p1 = fluid.flash_pt(P_L, cfg.T_cold)
        h1 = p1["h"]

        # State 2 estimate: comp1 outlet
        s2_is = fluid.flash_ps(P_H, p1["s"])
        h2 = h1 + (s2_is["h"] - h1) / cfg.eta_comp1
        p2 = fluid.flash_ph(P_H, h2)

        # State 5 estimate: turb in (P_H, T_hot)
        p5 = fluid.flash_pt(P_H, cfg.T_hot)
        h5 = p5["h"]

        # State 6 estimate: turb out (P_L)
        s6_is = fluid.flash_ps(P_L, p5["s"])
        h6 = h5 - (h5 - s6_is["h"]) * cfg.eta_turb
        p6 = fluid.flash_ph(P_L, h6)

        # Estimate mid temperatures
        T_mix = 0.5 * (p2["T"] + p6["T"])
        p_mix = fluid.flash_pt(P_H, T_mix)
        h_mix = p_mix["h"]

        T_mid_lp = 0.5 * (p6["T"] + cfg.T_cold + 50.0)
        p_mid_lp = fluid.flash_pt(P_L, T_mid_lp)
        h7 = p_mid_lp["h"]

        # LTR cold: h2 -> h_mix
        H_ltr_c = np.linspace(h2, h_mix, N)
        # LTR hot: h7 -> (h1 + 30 kJ/kg)
        h8 = h1 + 35000.0
        H_ltr_h = np.linspace(h8, h7, N)

        # HTR cold: h_mix -> (h5 - 150 kJ/kg)
        h4 = h5 - 160000.0
        H_htr_c = np.linspace(h_mix, h4, N)
        # HTR hot: h7 -> h6
        H_htr_h = np.linspace(h7, h6, N)

        # Initialize frozen capacities
        p_ltr_c_mid = fluid.flash_ph(P_H, float(np.mean(H_ltr_c)))
        p_ltr_h_mid = fluid.flash_ph(P_L, float(np.mean(H_ltr_h)))
        p_htr_c_mid = fluid.flash_ph(P_H, float(np.mean(H_htr_c)))
        p_htr_h_mid = fluid.flash_ph(P_L, float(np.mean(H_htr_h)))

        self.ltr_block.init_capacities(
            np.full(N, p_ltr_c_mid["rho"]), np.full(N, p_ltr_h_mid["rho"])
        )
        self.htr_block.init_capacities(
            np.full(N, p_htr_c_mid["rho"]), np.full(N, p_htr_h_mid["rho"])
        )

        Y0 = np.empty(self.n_states)
        Y0[self.sl_htr_c] = H_htr_c
        Y0[self.sl_htr_h] = H_htr_h
        Y0[self.sl_ltr_c] = H_ltr_c
        Y0[self.sl_ltr_h] = H_ltr_h
        Y0[self.idx_mix] = h_mix

        return Y0

    def build_jac_sparsity(self) -> csc_matrix:
        """
        Constructs the Jacobian sparsity pattern for solve_ivp BDF integrator.

        Two things govern the pattern:

        1. **Advection stencil.** Upwind couples a cell to its single upwind
           neighbour. MUSCL reconstructs each face from a limited slope, so a cell
           residual reaches two cells upwind and one cell downwind -- and, at a
           block inlet, the *second* cell also sees the inlet enthalpy (its upwind
           face carries the inlet cell's slope). Missing that entry is not benign:
           num_jac attributes a group's finite difference to the declared columns
           only, so an undeclared nonzero is folded into an unrelated entry.

        2. **Pressure chain.** Cell pressures march downstream, which would couple
           every cell to all of its upstream neighbours and make the pattern
           triangular-dense within each stream -- that alone is what drove the
           colouring to ~48 groups. RecuperatorBlock freezes the pressure profile
           whenever freeze_U is set, which is exactly the condition holding during
           every finite-difference column, so those couplings are genuinely absent
           from the perturbed residual and are not declared here.

        tests/test_jac_sparsity.py checks this pattern against a dense finite
        difference for both advection schemes.
        """
        N = self.N
        n_tot = self.n_states
        muscl = bool(self.cfg.use_muscl)
        S = lil_matrix((n_tot, n_tot), dtype=np.float64)

        def cold_stream(base: int, inlet_cols: list[int]) -> None:
            """Declares a cold stream (flows cell 0 -> N-1) occupying base..base+N-1.

            inlet_cols are the state columns the stream's inlet enthalpy depends on.
            """
            reach_up = 2 if muscl else 1     # cells upwind that the stencil reaches
            reach_down = 1 if muscl else 0   # cells downwind (via the limited slope)
            inlet_rows = 2 if muscl else 1   # rows that still see the inlet enthalpy
            for i in range(N):
                row = base + i
                for off in range(-reach_up, reach_down + 1):
                    j = i + off
                    if 0 <= j < N:
                        S[row, base + j] = 1.0
                S[row, base + N + i] = 1.0   # local hot cell, through Q_cell
                if i < inlet_rows:
                    for col in inlet_cols:
                        S[row, col] = 1.0

        def hot_stream(base: int, inlet_cols: list[int]) -> None:
            """Declares a hot stream (flows cell N-1 -> 0) occupying base..base+N-1.

            The hot inlet enters at cell N-1, so the inlet rows are at the top end.
            """
            reach_up = 2 if muscl else 1
            reach_down = 1 if muscl else 0
            inlet_rows = 2 if muscl else 1
            for i in range(N):
                row = base + i
                for off in range(-reach_down, reach_up + 1):
                    j = i + off
                    if 0 <= j < N:
                        S[row, base + j] = 1.0
                S[row, base - N + i] = 1.0   # local cold cell, through Q_cell
                if i >= N - inlet_rows:
                    for col in inlet_cols:
                        S[row, col] = 1.0

        # Column blocks: HTR_c 0..N-1, HTR_h N..2N-1, LTR_c 2N..3N-1, LTR_h 3N..4N-1.
        # Outlet enthalpies are cell values under upwind and extrapolated faces under
        # MUSCL, so an outlet reaches one cell further in when MUSCL is on.
        def cold_outlet_cols(base: int) -> list[int]:
            cols = [base + N - 1]
            if muscl and N > 1:
                cols.append(base + N - 2)
            return cols

        def hot_outlet_cols(base: int) -> list[int]:
            cols = [base]
            if muscl and N > 1:
                cols.append(base + 1)
            return cols

        # HTR cold inlet is the mixer outlet; HTR hot inlet is the turbine exit,
        # which traces back through the heater to the HTR cold outlet (H4).
        cold_stream(0, [self.idx_mix])
        hot_stream(N, cold_outlet_cols(0))

        # LTR cold inlet is the main-compressor exit. Under use_legacy_p_in its
        # pressure is fixed and under the alternative it comes from the previous
        # call's state, so it carries no live dependence on Y; the coupling to
        # State 8 is declared anyway because it is only two entries and it keeps
        # the pattern valid if that closure is ever made simultaneous.
        cold_stream(2 * N, [3 * N])
        # LTR hot inlet is the HTR hot outlet (State 7).
        hot_stream(3 * N, hot_outlet_cols(N))

        # Mixer row: blends the LTR cold outlet (H11) with the recompressor exit,
        # which is driven by State 8 = H_LTR_h[0] taken as a cell value, not a face.
        S[self.idx_mix, self.idx_mix] = 1.0
        for col in cold_outlet_cols(2 * N):
            S[self.idx_mix, col] = 1.0
        S[self.idx_mix, 3 * N] = 1.0

        return csc_matrix(S)

    def rhs(self, tau: float, Y: np.ndarray, freeze_U: bool = False) -> np.ndarray:
        """
        Master ODE RHS: dY/dtau = master_rhs(tau, Y).
        """
        N = self.N
        cfg = self.cfg
        fluid = self.fluid

        # Unpack state vector
        H_htr_c = Y[self.sl_htr_c]
        H_htr_h = Y[self.sl_htr_h]
        H_ltr_c = Y[self.sl_ltr_c]
        H_ltr_h = Y[self.sl_ltr_h]
        H_mix = float(Y[self.idx_mix])

        P_L = self.P_L
        P_H = cfg.P_H

        # ---------------------------------------------------------------------
        # Cycle Evaluation Pass
        # ---------------------------------------------------------------------
        # State 8 (LTR hot outlet)
        H8 = H_ltr_h[0]

        # In pseudo-transient march, determine State 1 & Compressor 1
        # P_1 is P_L if use_legacy_p_in, else P_8 (from previous LTR state if available)
        P1 = P_L if cfg.use_legacy_p_in else (
            self.last_cycle_state.ltr_state.P_hot_out if self.last_cycle_state else P_L
        )

        # State 1 (Comp 1 inlet)
        h1 = fluid.flash_pt(P1, cfg.T_cold)["h"]

        # Compressor 1: State 1 -> State 2 (P_H)
        h2, T2, w_comp1, p2_props = solve_compressor(
            P1, h1, P_H, cfg.eta_comp1, fluid
        )

        # LTR hot inlet is State 7 (HTR hot outlet)
        H7 = H_htr_h[0]
        P7 = (
            self.last_cycle_state.htr_state.P_hot_out if self.last_cycle_state else P_L
        )

        # Solve LTR Block:
        # Cold: flows 0 -> N-1 from (h2, P_H)
        # Hot:  flows N-1 -> 0 from (H7, P7)
        dH_ltr_c, dH_ltr_h, ltr_state = self.ltr_block.compute_residuals(
            H_ltr_c, H_ltr_h, h2, P_H, H7, P7, fluid, freeze_U=freeze_U
        )

        # State 11 (LTR cold outlet)
        H11 = ltr_state.H_cold_out
        P11 = ltr_state.P_cold_out

        # State 8 actual pressure from LTR
        P8 = ltr_state.P_hot_out

        # Recompressor (Comp 2): State 8 -> State 3 (P11)
        h3, T3, w_comp2, p3_props = solve_compressor(
            P8, H8, P11, cfg.eta_comp2, fluid
        )

        # Mixing junction: blends 11 (cold branch) + 3 (recomp branch)
        H_mix_target = solve_mixer(
            H_cold_branch=H11,
            m_dot_cold=self.flows.m_dot_cold,
            H_recomp_branch=h3,
            m_dot_recomp=self.flows.m_dot_recomp,
        )
        dH_mix = (H_mix_target - H_mix) / self.tau_mix

        # State 12 (HTR cold inlet): H_mix, P12 = P11
        P12 = P11

        # HTR Cold exit is State 4
        H4 = H_htr_c[-1]
        P4 = (
            self.last_cycle_state.htr_state.P_cold_out if self.last_cycle_state else P12
        )

        # Heater: State 4 -> State 5 (P4, T_hot)
        h5, Q_dot_in, p5_props = solve_heater(
            P4, H4, cfg.T_hot, self.flows.m_dot_hot, fluid
        )

        # Turbine: State 5 -> State 6 (P_L)
        h6, T6, w_turb, p6_props = solve_turbine(
            P4, h5, P_L, cfg.eta_turb, fluid
        )

        # Solve HTR Block:
        # Cold: flows 0 -> N-1 from (H_mix, P12)
        # Hot:  flows N-1 -> 0 from (h6, P_L)
        dH_htr_c, dH_htr_h, htr_state = self.htr_block.compute_residuals(
            H_htr_c, H_htr_h, H_mix, P12, h6, P_L, fluid, freeze_U=freeze_U
        )

        # ---------------------------------------------------------------------
        # Assemble Cycle Metrics and Cache State
        # ---------------------------------------------------------------------
        W_dot_turb = self.flows.m_dot_hot * w_turb
        W_dot_comp_1 = self.flows.m_dot_cold * w_comp1
        W_dot_comp_2 = self.flows.m_dot_recomp * w_comp2
        W_dot_net = W_dot_turb - W_dot_comp_1 - W_dot_comp_2

        # Pre-cooler heat rejection: State 8 -> State 1
        _, Q_dot_out, _ = solve_cooler(
            P8, H8, cfg.T_cold, self.flows.m_dot_cold, fluid, P_eval=P1
        )

        eta = W_dot_net / Q_dot_in if Q_dot_in > 0 else 0.0
        energy_balance_err = abs(Q_dot_in - Q_dot_out - W_dot_net)

        # Package stations dictionary.
        # Recuperator outlet enthalpies are cell-centre values under upwind but
        # extrapolated face values under MUSCL, so pairing them with the terminal
        # cell's temperature would report a (T, h) pair that is not a real state.
        # Re-flash instead.
        T4 = fluid.flash_ph(htr_state.P_cold_out, htr_state.H_cold_out)["T"]
        T7 = fluid.flash_ph(htr_state.P_hot_out, htr_state.H_hot_out)["T"]
        T8 = fluid.flash_ph(ltr_state.P_hot_out, ltr_state.H_hot_out)["T"]
        T11 = fluid.flash_ph(ltr_state.P_cold_out, ltr_state.H_cold_out)["T"]

        stations = {
            1: {"P": P1, "h": h1, "T": cfg.T_cold, "m_dot": self.flows.m_dot_cold},
            2: {"P": P_H, "h": h2, "T": T2, "m_dot": self.flows.m_dot_cold},
            3: {"P": P11, "h": h3, "T": T3, "m_dot": self.flows.m_dot_recomp},
            4: {"P": htr_state.P_cold_out, "h": htr_state.H_cold_out, "T": T4, "m_dot": self.flows.m_dot_hot},
            5: {"P": htr_state.P_cold_out, "h": h5, "T": cfg.T_hot, "m_dot": self.flows.m_dot_hot},
            6: {"P": P_L, "h": h6, "T": T6, "m_dot": self.flows.m_dot_hot},
            7: {"P": htr_state.P_hot_out, "h": htr_state.H_hot_out, "T": T7, "m_dot": self.flows.m_dot_hot},
            8: {"P": ltr_state.P_hot_out, "h": ltr_state.H_hot_out, "T": T8, "m_dot": self.flows.m_dot_hot},
            9: {"P": P8, "h": H8, "T": ltr_state.T_hot[0], "m_dot": self.flows.m_dot_recomp},
            10: {"P": P8, "h": H8, "T": ltr_state.T_hot[0], "m_dot": self.flows.m_dot_cold},
            11: {"P": ltr_state.P_cold_out, "h": ltr_state.H_cold_out, "T": T11, "m_dot": self.flows.m_dot_cold},
            12: {"P": P12, "h": H_mix, "T": fluid.flash_ph(P12, H_mix)["T"], "m_dot": self.flows.m_dot_hot},
        }

        self.last_cycle_state = FlowsheetCycleState(
            stations=stations,
            W_dot_turb=W_dot_turb,
            W_dot_comp_1=W_dot_comp_1,
            W_dot_comp_2=W_dot_comp_2,
            W_dot_net=W_dot_net,
            Q_dot_in=Q_dot_in,
            Q_dot_out=Q_dot_out,
            eta=eta,
            energy_balance_err=energy_balance_err,
            ltr_state=ltr_state,
            htr_state=htr_state,
        )

        # Assemble full RHS vector
        dY = np.empty_like(Y)
        dY[self.sl_htr_c] = dH_htr_c
        dY[self.sl_htr_h] = dH_htr_h
        dY[self.sl_ltr_c] = dH_ltr_c
        dY[self.sl_ltr_h] = dH_ltr_h
        dY[self.idx_mix] = dH_mix

        return dY
