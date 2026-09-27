"""
solver.py
=========
Pseudo-transient continuation (PTC) solver for the sCO2 recompression cycle.
Uses SciPy's stiff BDF integrator with sparse Jacobian coloring and adaptive chunking.
"""

from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Optional, Dict, Any, List
import numpy as np
from scipy.integrate import solve_ivp

from .config import CycleConfig, CycleDOF
from .thermo import FluidProperties
from .flowsheet import Flowsheet, FlowsheetCycleState


@dataclass
class CycleResult:
    """Results of the converged PTC cycle simulation."""
    converged: bool
    tau_final: float
    max_residual: float
    wall_time_s: float
    n_chunks: int
    n_fev: int
    y_final: np.ndarray
    state: FlowsheetCycleState
    dof: CycleDOF
    config: CycleConfig
    res_history: List[float]
    tau_history: List[float]
    # Property-window clamp counts accumulated during this solve. Non-zero means
    # flash_ph saturated some inputs, so part of the residual surface is flat and
    # "converged" may mean "converged against the clamp".
    n_clamped_P: int = 0
    n_clamped_h: int = 0

    @property
    def clamped(self) -> bool:
        return (self.n_clamped_P + self.n_clamped_h) > 0

    @property
    def eta(self) -> float:
        return self.state.eta

    @property
    def W_dot_net(self) -> float:
        return self.state.W_dot_net

    @property
    def W_dot_turb(self) -> float:
        return self.state.W_dot_turb

    @property
    def Q_dot_in(self) -> float:
        return self.state.Q_dot_in

    def summary(self) -> str:
        """Returns formatted string summarizing key performance metrics."""
        s = self.state
        st = s.stations
        lines = [
            "=" * 68,
            f"  sCO2 Recompression Brayton Cycle Result ({'CONVERGED' if self.converged else 'FAILED'})",
            "=" * 68,
            f"  Thermal Efficiency (eta)   : {self.eta * 100:.3f} %",
            f"  Net Power (W_dot_net)      : {self.W_dot_net / 1e6:.4f} MW",
            f"  Turbine Power (W_turb)     : {self.W_dot_turb / 1e6:.4f} MW",
            f"  Compressor 1 Power (W_c1)  : {s.W_dot_comp_1 / 1e6:.4f} MW",
            f"  Recompressor Power (W_c2)  : {s.W_dot_comp_2 / 1e6:.4f} MW",
            f"  Heat Addition (Q_dot_in)   : {self.Q_dot_in / 1e6:.4f} MW",
            f"  Heat Rejection (Q_dot_out) : {s.Q_dot_out / 1e6:.4f} MW",
            f"  Energy Balance Error       : {s.energy_balance_err:.2f} W",
            "-" * 68,
            f"  HTR Duty                   : {s.htr_state.Q_total / 1e6:.4f} MW",
            f"  LTR Duty                   : {s.ltr_state.Q_total / 1e6:.4f} MW",
            f"  HTR Min Pinch dT           : {s.htr_state.dT_min:.2f} K",
            f"  LTR Min Pinch dT           : {s.ltr_state.dT_min:.2f} K",
            f"  HTR HP Pressure Drop       : {s.htr_state.dP_cold / 1e5:.4f} bar",
            f"  HTR LP Pressure Drop       : {s.htr_state.dP_hot / 1e5:.4f} bar",
            f"  LTR HP Pressure Drop       : {s.ltr_state.dP_cold / 1e5:.4f} bar",
            f"  LTR LP Pressure Drop       : {s.ltr_state.dP_hot / 1e5:.4f} bar",
            "-" * 68,
            f"  Station 1 (Comp1 in)       : T={st[1]['T']:.2f} K, P={st[1]['P']/1e5:.3f} bar",
            f"  Station 2 (Comp1 out)      : T={st[2]['T']:.2f} K, P={st[2]['P']/1e5:.3f} bar",
            f"  Station 3 (Comp2 out)      : T={st[3]['T']:.2f} K, P={st[3]['P']/1e5:.3f} bar",
            f"  Station 4 (HTR HP out)     : T={st[4]['T']:.2f} K, P={st[4]['P']/1e5:.3f} bar",
            f"  Station 5 (Turb in)        : T={st[5]['T']:.2f} K, P={st[5]['P']/1e5:.3f} bar",
            f"  Station 6 (Turb out)       : T={st[6]['T']:.2f} K, P={st[6]['P']/1e5:.3f} bar",
            f"  Station 7 (HTR LP out)     : T={st[7]['T']:.2f} K, P={st[7]['P']/1e5:.3f} bar",
            f"  Station 8 (LTR LP out)     : T={st[8]['T']:.2f} K, P={st[8]['P']/1e5:.3f} bar",
            f"  Station 11 (LTR HP out)    : T={st[11]['T']:.2f} K, P={st[11]['P']/1e5:.3f} bar",
            f"  Station 12 (Mixer out)     : T={st[12]['T']:.2f} K, P={st[12]['P']/1e5:.3f} bar",
            "-" * 68,
            f"  Solver Wall Time           : {self.wall_time_s:.2f} s",
            f"  Final tau                  : {self.tau_final:.2e} s",
            f"  Max Residual               : {self.max_residual:.3e} J/(kg s)",
            "=" * 68,
        ]
        return "\n".join(lines)


def solve_cycle_ptc(
    dof: Optional[CycleDOF] = None,
    cfg: Optional[CycleConfig] = None,
    fluid: Optional[FluidProperties] = None,
    y0: Optional[np.ndarray] = None,
    tau_max: float = 1e4,
    chunk_dtau: float = 25.0,
    conv_tol: float = 5.0,
    rtol: float = 1e-4,
    atol: float = 1e2,
    adaptive_tol: bool = False,
    verbose: bool = True,
    max_wall_s: Optional[float] = None
) -> CycleResult:
    """
    Solves the complete supercritical cycle via pseudotransient continuation.

    Parameters:
    -----------
    dof : CycleDOF, optional (defaults to baseline r=0.5, frac_recomp=0.4, A_tube=4.2)
    cfg : CycleConfig, optional
    fluid : FluidProperties, optional
    y0 : np.ndarray, optional (warm start state vector)
    tau_max : float, maximum pseudo-time [s]
    chunk_dtau : float, duration of each integration chunk [s]
    conv_tol : float, residual convergence tolerance [J/(kg s)]
    """
    if cfg is None:
        cfg = CycleConfig()
    if dof is None:
        dof = CycleDOF()
    if fluid is None:
        fluid = FluidProperties(cfg.fluid_components, cfg.mole_fractions)

    t_start = time.time()
    flowsheet = Flowsheet(cfg, dof, fluid)
    clamp_P0, clamp_h0 = fluid.n_clamped_P, fluid.n_clamped_h

    if y0 is None:
        if verbose:
            print("  Initializing state vector from thermodynamic estimates...")
        y = flowsheet.build_initial_guess()
    else:
        if len(y0) != flowsheet.n_states:
            # Interpolate if resolution differs
            old_N = (len(y0) - 1) // 4
            new_N = dof.N_cells
            if verbose:
                print(f"  Interpolating warm-start vector from N={old_N} to N={new_N}...")
            y = np.empty(flowsheet.n_states)
            x_old = np.linspace(0, 1, old_N)
            x_new = np.linspace(0, 1, new_N)
            for s_idx in range(4):
                y[s_idx * new_N : (s_idx + 1) * new_N] = np.interp(
                    x_new, x_old, y0[s_idx * old_N : (s_idx + 1) * old_N]
                )
            y[-1] = y0[-1]
        else:
            y = y0.copy()

    # Initial residual check
    fluid.clear_cell_cache()
    r0 = flowsheet.rhs(0.0, y, freeze_U=False)
    max_res0 = float(np.max(np.abs(r0)))
    if verbose:
        print("=" * 68)
        print(f"  Starting PTC solve: N={dof.N_cells} (total states={flowsheet.n_states})")
        print(f"  DOF: r={dof.r:.3f}, frac_recomp={dof.frac_recomp:.3f}, A_tube={dof.A_tube:.2f} m^2")
        print(f"  Initial max residual: {max_res0:.3e} J/(kg s)")
        print(f"  {'tau':>10}  {'max_residual':>14}  {'elapsed':>10}")
        print("-" * 68)

    # State tracking for lazy HTC caching during num_jac and early-stage continuation
    last_t_eval = [-1.0]
    force_continuous = False

    def ode_system(t: float, y_vec: np.ndarray) -> np.ndarray:
        # If t is identical to last call, we are inside a finite difference perturbation column
        is_perturbation = (t == last_t_eval[0])
        last_t_eval[0] = t
        # Early-stage lazy refresh: freeze U during chunk only if residual > 40 AND no oscillation detected
        # If oscillation detected or residual <= 40, use continuous refresh
        freeze = is_perturbation or (max_res > 40.0 and not force_continuous)
        return flowsheet.rhs(t, y_vec, freeze_U=freeze)

    tau = 0.0
    n_chunks = 0
    total_fev = 0
    max_res = max_res0
    converged = False
    res_history = [max_res0]
    tau_history = [0.0]

    while tau < tau_max:
        chunk_end = min(tau + chunk_dtau, tau_max)
        n_chunks += 1

        if adaptive_tol:
            # Calibrated tolerances: respect physical cell-to-cell enthalpy gradients (Delta_h ~ 5 kJ/kg)
            # To avoid unphysical spatial perturbations, rtol must not exceed ~5e-4 (error < 500 J/kg)
            if max_res > 50.0:
                cur_rtol = min(rtol * 5.0, 5e-4)
                cur_atol = min(atol * 2.0, 2e2)
            elif max_res > 15.0:
                cur_rtol = min(rtol * 2.0, 2e-4)
                cur_atol = min(atol * 1.5, 1.5e2)
            else:
                cur_rtol = rtol
                cur_atol = atol
        else:
            cur_rtol = rtol
            cur_atol = atol

        sol = solve_ivp(
            ode_system,
            t_span=(tau, chunk_end),
            y0=y,
            method="BDF",
            jac_sparsity=flowsheet.jac_sparsity,
            rtol=cur_rtol,
            atol=cur_atol,
            max_step=15.0,
        )

        total_fev += sol.nfev
        if not sol.success:
            if verbose:
                print(f"  Warning: solve_ivp ended chunk with message: {sol.message}")
            break

        y = sol.y[:, -1]
        tau = float(sol.t[-1])

        # Evaluate full fresh residual with un-frozen U
        fluid.clear_cell_cache()
        r = flowsheet.rhs(tau, y, freeze_U=False)
        max_res = float(np.max(np.abs(r)))

        # Oscillation check: if residual bounced upwards across a chunk boundary,
        # permanently lock into continuous refresh mode to damp any lag
        if len(res_history) >= 2 and max_res > res_history[-1]:
            if not force_continuous:
                force_continuous = True
                if verbose:
                    print(f"  [Oscillation Detected] Residual bounced ({res_history[-1]:.1e} -> {max_res:.1e}). Switching permanently to continuous HTC refresh.")

        res_history.append(max_res)
        tau_history.append(tau)

        elapsed = time.time() - t_start
        if verbose:
            print(f"  {tau:>10.1f}  {max_res:>14.3e}  {elapsed:>9.1f} s", flush=True)

        if max_res < conv_tol:
            converged = True
            break

        if max_wall_s is not None and elapsed >= max_wall_s:
            if verbose:
                print(f"  Wall time limit of {max_wall_s:.1f} s reached.")
            break

        # Dynamically scale chunk duration as solution settles
        if max_res < 50.0:
            chunk_dtau = min(chunk_dtau * 1.5, 200.0)

    wall_time_s = time.time() - t_start
    n_clamped_P = fluid.n_clamped_P - clamp_P0
    n_clamped_h = fluid.n_clamped_h - clamp_h0
    if verbose:
        print("-" * 68)
        print(f"  Solve completed in {wall_time_s:.2f} s ({'Converged' if converged else 'Unconverged'})")
        if n_clamped_P or n_clamped_h:
            print(
                f"  WARNING: property window clamped {n_clamped_P} times on P and "
                f"{n_clamped_h} times on h - part of the residual surface is flat."
            )

    return CycleResult(
        converged=converged,
        tau_final=tau,
        max_residual=max_res,
        wall_time_s=wall_time_s,
        n_chunks=n_chunks,
        n_fev=total_fev,
        y_final=y,
        state=flowsheet.last_cycle_state,
        dof=dof,
        config=cfg,
        res_history=res_history,
        tau_history=tau_history,
        n_clamped_P=n_clamped_P,
        n_clamped_h=n_clamped_h,
    )


def solve_cycle_ladder(
    target_dof: CycleDOF,
    cfg: Optional[CycleConfig] = None,
    fluid: Optional[FluidProperties] = None,
    ladder: Optional[List[int]] = None,
    conv_tol: float = 2.0,
    adaptive_tol: bool = False,
    verbose: bool = True,
) -> CycleResult:
    """
    Multigrid resolution ladder solver (DynamicHX pattern):
    Solves coarser grids to loose tolerance to establish gross temperature/pressure
    profiles rapidly, then successively warm-starts finer grids.

    Default ladder: 10 -> 20 -> 40 -> target_N
    """
    if cfg is None:
        cfg = CycleConfig()
    if fluid is None:
        fluid = FluidProperties(cfg.fluid_components, cfg.mole_fractions)

    target_N = target_dof.N_cells
    if ladder is None:
        if target_N >= 80:
            ladder = [10, 20, 40, target_N]
        elif target_N >= 40:
            ladder = [10, 20, target_N]
        elif target_N >= 20:
            ladder = [10, target_N]
        else:
            ladder = [target_N]

    t_ladder_start = time.time()
    if verbose:
        print("=" * 68)
        print(f"  Starting Multigrid Ladder Solve: {' -> '.join(f'N={n}' for n in ladder)}")
        print("=" * 68)

    warm_y: Optional[np.ndarray] = None
    res: Optional[CycleResult] = None

    for stage_idx, n_stage in enumerate(ladder):
        is_final = (stage_idx == len(ladder) - 1)
        stage_tol = conv_tol if is_final else max(conv_tol * 4.0, 15.0)
        stage_tau_max = 500.0 if not is_final else 800.0

        stage_dof = CycleDOF(
            A_tube=target_dof.A_tube,
            r=target_dof.r,
            frac_recomp=target_dof.frac_recomp,
            N_cells=n_stage,
        )

        if verbose:
            print(f"\n  --- Ladder Stage {stage_idx + 1}/{len(ladder)}: N={n_stage} (tol={stage_tol:.1f}) ---")

        res = solve_cycle_ptc(
            dof=stage_dof,
            cfg=cfg,
            fluid=fluid,
            y0=warm_y,
            tau_max=stage_tau_max,
            chunk_dtau=25.0,
            conv_tol=stage_tol,
            adaptive_tol=adaptive_tol,
            verbose=verbose,
        )
        warm_y = res.y_final

    total_ladder_time = time.time() - t_ladder_start
    if verbose:
        print("=" * 68)
        print(f"  Ladder Solve Finished in {total_ladder_time:.2f} s Total")
        print("=" * 68)

    return res
