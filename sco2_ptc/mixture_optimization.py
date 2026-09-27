"""
mixture_optimization.py
=======================
Multi-Worker Multi-Component Mixture Optimization for Supercritical Brayton Cycles.

Evaluates 6 working fluid compositions:
  1. Pure sCO2 (100% CO2, P_L = 83.8 bar)
  2. 95% CO2 + 5% Ethane (P_L = 81.7 bar)
  3. 91% CO2 + 9% Ethane (P_L = 79.5 bar, baseline)
  4. 85% CO2 + 15% Ethane (P_L = 76.7 bar)
  5. 80% CO2 + 20% Ethane (P_L = 74.3 bar)
  6. 75% CO2 + 25% Ethane (P_L = 72.3 bar)

Each mixture is assigned to an independent parallel worker process.
Within each worker:
  - Optimizes (r*, frac_recomp*) to +-0.01 (1% point) tolerance using 2D Nelder-Mead.
  - Leverages continuous warm-starting for rapid 5-15s evaluation cycles.
  - Writes progress directly to its own unbuffered log file (worker_<name>.log).
  - Evaluates full post-convergence Second-Law exergy destruction.

The master process compiles `mixture_optima_summary.csv` and generates publication plots.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
import time
import csv
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Configure DLL directory for conda environment to ensure BLAS/LAPACK DLLs load in worker processes
conda_prefix = Path(sys.executable).parent
for sub in ["Library/bin", "Library/usr/bin", "Library/mingw-w64/bin", "Scripts"]:
    p_sub = conda_prefix / sub
    if p_sub.exists():
        os.environ["PATH"] = str(p_sub) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(p_sub))
            except Exception:
                pass

# Ensure parent python directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, field

from sco2_ptc.config import CycleConfig, CycleDOF
from sco2_ptc.thermo import FluidProperties
from sco2_ptc.solver import solve_cycle_ptc, CycleResult
from sco2_ptc.exergy import compute_cycle_exergy, CycleExergyAnalysis
from sco2_ptc.io import save_cycle_result, load_cycle_result
from sco2_ptc.fluids import CAMPAIGN_FLUIDS, WorkingFluid, critical_pressure, paradigm_pressures


@dataclass(frozen=True)
class OptimizationTask:
    """One fluid-at-one-pressure-condition optimisation job.

    This used to be a bare positional tuple whose length (15 versus 16) decided
    whether P_H was supplied or silently defaulted to 220 bar. Adding a field to
    either call site would have rerouted the unpacking without any error, so the
    protocol is now explicit. Frozen and picklable, for ProcessPoolExecutor.
    """
    name: str
    components: List[str]
    mole_fractions: List[float]
    p_L: float
    p_H: float
    save_dir: str
    N_cells: int = 40
    conv_tol: float = 2.0
    use_muscl: bool = True
    muscl_limiter: str = "van_leer"
    r_init: float = 0.50
    frac_init: float = 0.35
    r_bounds: Tuple[float, float] = (0.25, 0.75)
    frac_bounds: Tuple[float, float] = (0.15, 0.55)
    xatol: float = 0.01
    max_iter: int = 25
    grid_size: int = 10

    @property
    def slug(self) -> str:
        return (
            self.name.lower().replace(" ", "_").replace("%", "pct").replace("+", "")
        )


# Campaign fluid set lives in sco2_ptc.fluids so that this module and
# run_three_paradigms.py cannot drift apart again.
# This entry point is reserved for CO2/ethane mixtures. Pure CO2 has its own
# standalone entry point in pure_co2_optimization.py.
DEFAULT_MIXTURES = CAMPAIGN_FLUIDS[1:]


def optimize_single_mixture_worker(task: "OptimizationTask | Tuple") -> Dict[str, Any]:
    """
    Independent worker function to optimize a single mixture composition.
    Writes unbuffered progress to a dedicated log file.
    """
    if not isinstance(task, OptimizationTask):
        raise TypeError(
            "optimize_single_mixture_worker now takes an OptimizationTask; the "
            "positional-tuple protocol was removed because its length decided how "
            "the fields were interpreted."
        )
    name = task.name
    fluid_components = task.components
    mole_fractions = task.mole_fractions
    p_L = task.p_L
    p_H = task.p_H
    N_cells = task.N_cells
    conv_tol = task.conv_tol
    use_muscl = task.use_muscl
    muscl_limiter = task.muscl_limiter
    r_init = task.r_init
    frac_init = task.frac_init
    r_bounds = task.r_bounds
    frac_bounds = task.frac_bounds
    xatol = task.xatol
    max_iter = task.max_iter
    grid_size = task.grid_size
    save_dir_str = task.save_dir

    save_dir = Path(save_dir_str)
    clean_name = name.lower().replace(" ", "_").replace("%", "pct").replace("+", "")
    log_file = save_dir / f"worker_{clean_name}.log"

    def log(msg: str):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] [{name}] {msg}"
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
        print(line, flush=True)

    # 0. Check if an optimization archive already exists (completed run)
    opt_archive = save_dir / f"opt_{clean_name}.npz"
    if opt_archive.exists():
        try:
            meta, arrays = load_cycle_result(opt_archive)
            log(f"Found existing completed optimization archive {opt_archive.name}. Returning cached results.")
            perf = meta.get("performance", {})
            dof = meta.get("dof", {})
            ex = meta.get("exergy", {})
            comps_ex = ex.get("components", {})
            fluid = FluidProperties(fluid_components, mole_fractions)
            summary = {
                "name": name,
                "x_CO2": mole_fractions[0],
                "x_second": mole_fractions[1] if len(mole_fractions) > 1 else 0.0,
                "T_crit": fluid.T_crit,
                "P_crit": fluid.P_crit / 1e5,
                "P_L_bar": p_L / 1e5,
                "P_H_bar": p_H / 1e5,
                "pressure_ratio": (p_H / p_L),
                "r_opt": dof.get("r", r_init),
                "frac_opt": dof.get("frac_recomp", frac_init),
                "r_opt_raw": dof.get("r", r_init),
                "frac_opt_raw": dof.get("frac_recomp", frac_init),
                "eta_best_seen": perf.get("eta", 0.0) * 100.0,
                "converged_final": bool(meta.get("converged", True)),
                "max_residual_final": meta.get("max_residual", float("nan")),
                "eta": perf.get("eta", 0.0) * 100.0,
                "W_net_MW": perf.get("W_dot_net", 0.0) / 1e6,
                "W_turb_MW": perf.get("W_dot_turb", 0.0) / 1e6,
                "W_c1_MW": perf.get("W_dot_comp_1", 0.0) / 1e6,
                "W_c2_MW": perf.get("W_dot_comp_2", 0.0) / 1e6,
                "Q_in_MW": perf.get("Q_dot_in", 0.0) / 1e6,
                "htr_dT_min": meta.get("recuperators", {}).get("HTR", {}).get("dT_min", 0.0),
                "ltr_dT_min": meta.get("recuperators", {}).get("LTR", {}).get("dT_min", 0.0),
                "S_gen_HTR": comps_ex.get("HTR Recuperator", {}).get("S_dot_gen", 0.0),
                "S_gen_LTR": comps_ex.get("LTR Recuperator", {}).get("S_dot_gen", 0.0),
                "S_gen_Heater": comps_ex.get("Primary Heater", {}).get("S_dot_gen", 0.0),
                "S_gen_Cooler": comps_ex.get("Pre-Cooler", {}).get("S_dot_gen", 0.0),
                "S_gen_Turb": comps_ex.get("Turbine", {}).get("S_dot_gen", 0.0),
                "S_gen_Comp1": comps_ex.get("Main Compressor 1", {}).get("S_dot_gen", 0.0),
                "S_gen_Comp2": comps_ex.get("Recompressor 2", {}).get("S_dot_gen", 0.0),
                "S_gen_Total": ex.get("S_dot_gen_total", 0.0),
                "exergy_eta": ex.get("eta_exergetic", 0.0) * 100.0,
                "evals": meta.get("n_fev", 0),
                "wall_time_s": meta.get("wall_time_s", 0.0),
            }
            return summary
        except Exception as e_arc:
            log(f"Notice: Could not deserialize existing {opt_archive.name} ({e_arc}), re-optimizing.")

    # Check for existing evaluations in history CSV to support resume & append
    history_file = save_dir / f"history_{clean_name}.csv"
    existing_evals = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as hf:
                reader = csv.DictReader(hf)
                for row in reader:
                    if row.get("eval_idx") and row.get("eta_th_pct"):
                        existing_evals.append(row)
        except Exception:
            existing_evals = []

    resuming = len(existing_evals) > 0

    # Initialize / append log file
    log_mode = "a" if (log_file.exists() and resuming) else "w"
    with open(log_file, log_mode, encoding="utf-8") as fh:
        if resuming:
            fh.write(f"\n=== Resuming Optimization Worker for {name} (PID: {os.getpid()}) ===\n")
        else:
            fh.write(f"=== Optimization Worker for {name} (PID: {os.getpid()}) ===\n")
        fh.flush()

    if not history_file.exists() or not resuming:
        with open(history_file, "w", newline="", encoding="utf-8") as hf:
            hwriter = csv.writer(hf)
            hwriter.writerow([
                "eval_idx", "r", "frac_recomp", "eta_th_pct", "W_net_MW",
                "htr_dT_min_K", "ltr_dT_min_K", "max_residual", "converged", "wall_time_s"
            ])
            hf.flush()

    log(f"Starting worker on PID {os.getpid()} | Grid N={N_cells} | MUSCL={use_muscl} ({muscl_limiter})")
    log(f"Components: {fluid_components} | Mole Fractions: {mole_fractions} | P_L = {p_L/1e5:.2f} bar")

    # 1. Initialize fluid
    fluid = FluidProperties(fluid_components, mole_fractions)
    log(f"Fluid Critical Point: T_crit = {fluid.T_crit:.2f} K, P_crit = {fluid.P_crit/1e5:.2f} bar")

    # 2. Base configuration
    cfg = CycleConfig(
        use_legacy_p_in=True,
        P_L_override=p_L,
        P_H=p_H,
        fluid_components=fluid_components,
        x_CO2=mole_fractions[0],
        use_muscl=use_muscl,
        muscl_limiter=muscl_limiter,
    )

    cache: Dict[Tuple[float, float], np.ndarray] = {}
    last_y = [None]
    eval_count = [0]
    best_eta = [-1.0]
    best_score = [np.inf]
    best_data = [None]
    last_evaluation = [None]

    # If resuming, update eval_count and warm-start point from best recorded history evaluation
    if resuming:
        try:
            eval_count[0] = max(int(row["eval_idx"]) for row in existing_evals)
            # Only converged rows are eligible to seed the restart. An unconverged
            # solve can report a spuriously high efficiency -- the paradigm 1 logs
            # contain rows at eta 47.8% with a residual of 3807 -- and taking the
            # raw maximum would warm-start the restart from exactly those points.
            converged_evals = [
                row for row in existing_evals
                if str(row.get("converged", "")).strip().lower() == "true"
            ]
            if converged_evals:
                best_row = max(converged_evals, key=lambda row: float(row["eta_th_pct"]))
                best_eta[0] = float(best_row["eta_th_pct"]) / 100.0
                r_init = float(best_row["r"])
                frac_init = float(best_row["frac_recomp"])
                log(
                    f"Resuming with {len(existing_evals)} previous evaluations "
                    f"({len(converged_evals)} converged, last eval_idx={eval_count[0]}). "
                    f"Warm-starting from best converged point: r={r_init:.4f}, "
                    f"frac={frac_init:.4f} (prior best eta={best_eta[0]*100:.3f}%)"
                )
            else:
                log(
                    f"Resuming with {len(existing_evals)} previous evaluations, none of "
                    f"which converged. Ignoring them for warm-start and restarting the "
                    f"simplex at the supplied initial point (r={r_init:.4f}, frac={frac_init:.4f})."
                )
        except Exception as e_res:
            log(f"Warning parsing existing evaluations ({e_res}), starting at provided initial point.")

    # Time-limit for warmstarted solves (generous 800s limit as per user directive)
    TIME_LIMIT = 800.0
    checkpoint_file = save_dir / f"checkpoint_{clean_name}.npz"

    # Check for intermediate checkpoint file
    if checkpoint_file.exists():
        try:
            c_meta, c_arrays = load_cycle_result(checkpoint_file)
            last_y[0] = c_arrays["y_final"]
            cache[(round(r_init, 3), round(frac_init, 3))] = c_arrays["y_final"]
            log(f"Loaded warm-start state vector from checkpoint: {checkpoint_file.name}")
        except Exception as chk_err:
            log(f"Warning: Failed loading checkpoint ({chk_err})")

    # -------------------------------------------------------------------------
    # Initial Grid-Ladder Bootstrap at (r_init, frac_init): N=10 -> N=20 -> N_cells
    # Only performed if no checkpoint state is available
    # -------------------------------------------------------------------------
    if last_y[0] is None:
        log(f"Bootstrapping initial state via coarse grid-ladder at (r={r_init:.2f}, frac={frac_init:.2f}): N=10 -> N=20 -> N={N_cells}...")
        t_boot = time.time()
        try:
            # Step 1: Coarse N=10 solve (unconditionally converges from thermodynamic estimate)
            dof_10 = CycleDOF(A_tube=4.20, r=r_init, frac_recomp=frac_init, N_cells=10)
            res_10 = solve_cycle_ptc(
                dof=dof_10, cfg=cfg, fluid=fluid, y0=None,
                tau_max=300.0, chunk_dtau=25.0, conv_tol=max(conv_tol, 5.0), max_wall_s=150.0, verbose=False
            )

            # Step 2: Intermediate N=20 solve (warmstarted from interpolated N=10)
            dof_20 = CycleDOF(A_tube=4.20, r=r_init, frac_recomp=frac_init, N_cells=20)
            res_20 = solve_cycle_ptc(
                dof=dof_20, cfg=cfg, fluid=fluid, y0=res_10.y_final if res_10.converged else None,
                tau_max=400.0, chunk_dtau=25.0, conv_tol=conv_tol, max_wall_s=300.0, verbose=False
            )

            # Step 3: Target N=N_cells solve (warmstarted from interpolated N=20)
            dof_root = CycleDOF(A_tube=4.20, r=r_init, frac_recomp=frac_init, N_cells=N_cells)
            res_root = solve_cycle_ptc(
                dof=dof_root, cfg=cfg, fluid=fluid, y0=res_20.y_final if res_20.converged else None,
                tau_max=600.0, chunk_dtau=25.0, conv_tol=conv_tol, max_wall_s=800.0, verbose=False
            )

            boot_elapsed = time.time() - t_boot
            if res_root.converged:
                log(f"Grid-ladder bootstrap succeeded in {boot_elapsed:.1f}s (eta={res_root.eta*100:.2f}%)!")
                last_y[0] = res_root.y_final
                cache[(round(r_init, 3), round(frac_init, 3))] = res_root.y_final
                try:
                    save_cycle_result(res_root, checkpoint_file)
                except Exception:
                    pass
            else:
                log(f"Warning: Initial bootstrap reached limit ({boot_elapsed:.1f}s). Proceeding with best available state.")
                last_y[0] = res_root.y_final
        except Exception as boot_err:
            log(f"Warning: Grid bootstrap encountered error ({boot_err}); solver will cold-start.")

    def evaluate_ptc(r_val: float, f_val: float) -> Tuple[float, CycleResult]:
        key = (round(r_val, 3), round(f_val, 3))
        warm_start = cache.get(key, last_y[0])

        dof = CycleDOF(A_tube=4.20, r=r_val, frac_recomp=f_val, N_cells=N_cells)
        
        # Primary solve: Warmstart with TIME_LIMIT
        t0 = time.time()
        res = solve_cycle_ptc(
            dof=dof,
            cfg=cfg,
            fluid=fluid,
            y0=warm_start,
            tau_max=500.0,
            chunk_dtau=25.0,
            conv_tol=conv_tol,
            max_wall_s=TIME_LIMIT,
            verbose=False,
        )

        # Fallback: If not converged within TIME_LIMIT, execute Grid-Ladder recovery
        if not res.converged:
            log(f"  [Timeout/Divergence Fallback] Solve at (r={r_val:.3f}, frac={f_val:.3f}) exceeded {TIME_LIMIT:.1f}s. Triggering grid-ladder recovery...")
            try:
                # Step A: Coarse N=10
                dof_10 = CycleDOF(A_tube=4.20, r=r_val, frac_recomp=f_val, N_cells=10)
                res_10 = solve_cycle_ptc(
                    dof=dof_10, cfg=cfg, fluid=fluid, y0=None,
                    tau_max=200.0, chunk_dtau=25.0, conv_tol=max(conv_tol, 5.0), max_wall_s=150.0, verbose=False
                )
                # Step B: Intermediate N=20
                dof_20 = CycleDOF(A_tube=4.20, r=r_val, frac_recomp=f_val, N_cells=20)
                res_20 = solve_cycle_ptc(
                    dof=dof_20, cfg=cfg, fluid=fluid, y0=res_10.y_final if res_10.converged else None,
                    tau_max=300.0, chunk_dtau=25.0, conv_tol=conv_tol, max_wall_s=300.0, verbose=False
                )
                # Step C: Target N=N_cells
                res = solve_cycle_ptc(
                    dof=dof, cfg=cfg, fluid=fluid, y0=res_20.y_final if res_20.converged else None,
                    tau_max=600.0, chunk_dtau=25.0, conv_tol=conv_tol, max_wall_s=800.0, verbose=False
                )
                log(f"  [Grid-Ladder Recovery] Completed in {time.time() - t0:.1f}s: converged={res.converged}, eta={res.eta*100:.2f}%")
            except Exception as rec_err:
                log(f"  [Grid-Ladder Recovery Error] {rec_err}")

        if res.converged or (res.max_residual < 50.0):
            last_y[0] = res.y_final
            cache[key] = res.y_final
            try:
                save_cycle_result(res, checkpoint_file)
            except Exception:
                pass

        return res.eta, res

    def objective(x: np.ndarray) -> float:
        # Bounds are enforced by Nelder-Mead itself (see the `bounds` argument
        # below), which clips trial points instead of letting them fall off a
        # penalty cliff -- the old cliff was discontinuous against an objective of
        # scale eta ~ 0.46, and its evaluations were counted but never logged, so
        # eval_idx developed gaps.
        r_val = float(np.clip(x[0], r_bounds[0], r_bounds[1]))
        f_val = float(np.clip(x[1], frac_bounds[0], frac_bounds[1]))
        eval_count[0] += 1

        t0 = time.time()
        eta, res = evaluate_ptc(r_val, f_val)
        dt = time.time() - t0

        min_pinch = min(res.state.htr_state.dT_min, res.state.ltr_state.dT_min)
        pinch_pen = 0.0
        if min_pinch < 2.0:
            pinch_pen = 0.10 * (2.0 - min_pinch)

        score = -eta + pinch_pen
        last_evaluation[0] = (score, res, dt, r_val, f_val)

        # Track the best point on the same quantity Nelder-Mead minimises. Ranking
        # on raw eta while minimising a pinch-penalised score made the point logged
        # as "best" disagree with the point the optimiser was moving toward.
        is_acceptable = res.converged or (res.max_residual < 30.0)
        if is_acceptable and score < best_score[0]:
            best_score[0] = score
            best_eta[0] = eta
            best_data[0] = (r_val, f_val, eta, min_pinch)
            marker = " ** Best **"
        else:
            marker = ""

        log(
            f"Iter {eval_count[0]:>2d}: r={r_val:.3f}, frac={f_val:.3f} -> "
            f"eta={eta*100:.3f}% (pinch={min_pinch:.1f} K, res={res.max_residual:.1f}, {dt:.1f}s){marker}"
        )

        # Record evaluation point to history CSV
        try:
            with open(history_file, "a", newline="", encoding="utf-8") as hf:
                hwriter = csv.writer(hf)
                hwriter.writerow([
                    eval_count[0],
                    round(r_val, 4),
                    round(f_val, 4),
                    round(eta * 100.0, 4),
                    round(res.state.W_dot_net / 1e6, 4) if res.state else 0.0,
                    round(res.state.htr_state.dT_min, 3) if res.state else 0.0,
                    round(res.state.ltr_state.dT_min, 3) if res.state else 0.0,
                    round(res.max_residual, 2),
                    res.converged,
                    round(dt, 2),
                ])
                hf.flush()
        except Exception as e_hist:
            log(f"Warning: Failed writing history row: {e_hist}")

        return score

    # Estimate the objective surface on a regular grid before local optimization.
    grid_best = [None]
    grid_file = save_dir / f"grid_{clean_name}.csv"
    log(f"Starting {grid_size}x{grid_size} grid scan over r and frac_recomp before optimization...")
    with open(grid_file, "w", newline="", encoding="utf-8") as gf:
        grid_writer = csv.writer(gf)
        grid_writer.writerow([
            "grid_i", "grid_j", "r", "frac_recomp", "score", "eta_th_pct",
            "min_pinch_K", "max_residual", "converged", "wall_time_s"
        ])
        r_values = np.linspace(r_bounds[0], r_bounds[1], grid_size)
        frac_values = np.linspace(frac_bounds[0], frac_bounds[1], grid_size)
        efficiency_grid = np.full((grid_size, grid_size), np.nan)
        for grid_i, r_value in enumerate(r_values):
            for grid_j, frac_value in enumerate(frac_values):
                score = objective(np.array([r_value, frac_value]))
                _, grid_res, solve_dt, _, _ = last_evaluation[0]
                min_pinch = min(grid_res.state.htr_state.dT_min, grid_res.state.ltr_state.dT_min)
                efficiency_grid[grid_i, grid_j] = grid_res.eta * 100.0
                grid_writer.writerow([
                    grid_i,
                    grid_j,
                    round(float(r_value), 6),
                    round(float(frac_value), 6),
                    round(float(score), 8),
                    round(grid_res.eta * 100.0, 6),
                    round(min_pinch, 6),
                    round(grid_res.max_residual, 6),
                    grid_res.converged,
                    round(solve_dt, 3),
                ])
                gf.flush()
                if (grid_res.converged or grid_res.max_residual < 30.0) and (
                    grid_best[0] is None or score < grid_best[0][0]
                ):
                    grid_best[0] = (float(score), float(r_value), float(frac_value))

    grid_plot = save_dir / f"grid_efficiency_{clean_name}.png"
    try:
        plot_r, plot_frac = np.meshgrid(r_values, frac_values, indexing="ij")
        figure = plt.figure(figsize=(9, 7))
        axes = figure.add_subplot(111, projection="3d")
        surface = axes.plot_surface(
            plot_r,
            plot_frac,
            efficiency_grid,
            cmap="viridis",
            edgecolor="none",
            antialiased=True,
        )
        axes.set_xlabel("Recompression fraction r")
        axes.set_ylabel("Recompression fraction split")
        axes.set_zlabel("Thermal efficiency (%)")
        axes.set_title(f"Efficiency grid: {name}")
        figure.colorbar(surface, ax=axes, shrink=0.65, pad=0.1, label="Efficiency (%)")
        figure.tight_layout()
        figure.savefig(grid_plot, dpi=180, bbox_inches="tight")
        plt.close(figure)
        log(f"Saved grid efficiency visualization to {grid_plot.name}.")
    except Exception as plot_err:
        log(f"Warning: Could not generate grid efficiency plot ({plot_err}).")

    if grid_best[0] is not None:
        _, r_init, frac_init = grid_best[0]
        r_step = (r_bounds[1] - r_bounds[0]) / max(grid_size - 1, 1)
        frac_step = (frac_bounds[1] - frac_bounds[0]) / max(grid_size - 1, 1)
        local_r_bounds = (
            max(r_bounds[0], r_init - r_step),
            min(r_bounds[1], r_init + r_step),
        )
        local_frac_bounds = (
            max(frac_bounds[0], frac_init - frac_step),
            min(frac_bounds[1], frac_init + frac_step),
        )
        log(
            f"Grid scan selected initial point r={r_init:.4f}, frac={frac_init:.4f}; "
            f"local search bounds are r={local_r_bounds}, frac={local_frac_bounds}; "
            f"results saved to {grid_file.name}."
        )
    else:
        local_r_bounds = r_bounds
        local_frac_bounds = frac_bounds
        log("Grid scan found no acceptable point; using the supplied initial point.")

    # Optimize with Nelder-Mead terminating at xatol
    t_opt_start = time.time()
    x0 = np.array([r_init, frac_init])
    opt_res = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        bounds=[local_r_bounds, local_frac_bounds],
        options={
            "xatol": xatol,
            "fatol": 1e-4,
            "maxiter": max_iter,
            "disp": False,
        },
    )

    r_opt_raw, frac_opt_raw = float(opt_res.x[0]), float(opt_res.x[1])
    # Rounding to 2 dp quantises at the same order as xatol (0.01), so the reported
    # point is not exactly the point the optimiser found. Both are recorded.
    r_opt = round(r_opt_raw, 2)
    frac_opt = round(frac_opt_raw, 2)

    log(f"Optimization finished in {time.time() - t_opt_start:.1f}s ({eval_count[0]} evals).")
    log(f"Candidate optimum (raw): r* = {r_opt_raw:.4f}, frac_recomp* = {frac_opt_raw:.4f}")
    log(f"Candidate optimum (reported, rounded to xatol): r* = {r_opt:.2f}, frac_recomp* = {frac_opt:.2f}")
    if best_data[0] is not None:
        b_r, b_f, b_eta, b_pinch = best_data[0]
        log(f"Best point actually evaluated: r={b_r:.4f}, frac={b_f:.4f}, "
            f"eta={b_eta*100:.3f}%, min pinch={b_pinch:.2f} K")

    # Final post-convergence verification solve at rounded (r*, frac_recomp*)
    log("Running final verification solve and Second-Law exergy calculation...")
    t0 = time.time()
    final_eta, final_res = evaluate_ptc(r_opt, frac_opt)
    dt_final = time.time() - t0
    exergy = compute_cycle_exergy(final_res.state, fluid, T0=298.15)

    # Save converged .npz archive
    archive_path = save_dir / f"opt_{clean_name}.npz"
    save_cycle_result(final_res, archive_path, exergy=exergy)
    log(f"Saved converged archive to {archive_path.name} ({dt_final:.1f}s)")

    s_htr = exergy.components["HTR Recuperator"].S_dot_gen
    s_ltr = exergy.components["LTR Recuperator"].S_dot_gen
    s_heat = exergy.components["Primary Heater"].S_dot_gen
    s_cool = exergy.components["Pre-Cooler"].S_dot_gen
    s_turb = exergy.components["Turbine"].S_dot_gen
    s_c1 = exergy.components["Main Compressor 1"].S_dot_gen
    s_c2 = exergy.components["Recompressor 2"].S_dot_gen

    summary = {
        "name": name,
        "x_CO2": mole_fractions[0],
        "x_second": mole_fractions[1] if len(mole_fractions) > 1 else 0.0,
        "T_crit": fluid.T_crit,
        "P_crit": fluid.P_crit / 1e5,
        "P_L_bar": p_L / 1e5,
        "P_H_bar": p_H / 1e5,
        "pressure_ratio": (p_H / p_L),
        "r_opt": r_opt,
        "frac_opt": frac_opt,
        "r_opt_raw": round(r_opt_raw, 4),
        "frac_opt_raw": round(frac_opt_raw, 4),
        "eta_best_seen": (best_data[0][2] * 100.0) if best_data[0] is not None else float("nan"),
        "converged_final": bool(final_res.converged),
        "max_residual_final": final_res.max_residual,
        "eta": final_res.eta * 100.0,
        "W_net_MW": final_res.W_dot_net / 1e6,
        "W_turb_MW": final_res.W_dot_turb / 1e6,
        "W_c1_MW": final_res.state.W_dot_comp_1 / 1e6,
        "W_c2_MW": final_res.state.W_dot_comp_2 / 1e6,
        "Q_in_MW": final_res.Q_dot_in / 1e6,
        "htr_dT_min": final_res.state.htr_state.dT_min,
        "ltr_dT_min": final_res.state.ltr_state.dT_min,
        "S_gen_HTR": s_htr,
        "S_gen_LTR": s_ltr,
        "S_gen_Heater": s_heat,
        "S_gen_Cooler": s_cool,
        "S_gen_Turb": s_turb,
        "S_gen_Comp1": s_c1,
        "S_gen_Comp2": s_c2,
        "S_gen_Total": exergy.S_dot_gen_total,
        "exergy_eta": exergy.exergetic_efficiency * 100.0,
        "evals": eval_count[0],
        "wall_time_s": time.time() - t_opt_start,
    }
    log(f"All done! eta* = {summary['eta']:.3f}%, W_net* = {summary['W_net_MW']:.4f} MW")
    return summary


class MixtureOptimizer:
    """Manages sequential optimization across fluid compositions."""

    def __init__(
        self,
        N_cells: int = 40,
        conv_tol: float = 2.0,
        use_muscl: bool = True,
        muscl_limiter: str = "van_leer",
        n_workers: int = 6,
        save_dir: Optional[Path] = None,
        xatol: float = 0.01,
        max_iter: int = 25,
    ):
        self.N_cells = N_cells
        self.conv_tol = conv_tol
        self.use_muscl = use_muscl
        self.muscl_limiter = muscl_limiter
        self.n_workers = n_workers
        self.xatol = xatol
        self.max_iter = max_iter

        if save_dir is None:
            self.save_dir = Path(__file__).resolve().parent.parent.parent / "results" / "mixture_optimization"
        else:
            self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def run_all_mixtures(
        self,
        mixtures: Optional[List[Tuple[str, List[str], List[float], float]]] = None,
    ) -> List[Dict[str, Any]]:
        if mixtures is None:
            mixtures = DEFAULT_MIXTURES

        print("=" * 88, flush=True)
        print("  PARALLEL MULTI-COMPONENT MIXTURE OPTIMIZATION CAMPAIGN", flush=True)
        print(f"  Total Compositions: {len(mixtures)} | Sequential mode", flush=True)
        print(f"  Grid Dimension: N={self.N_cells} cells/stream | MUSCL: {self.use_muscl} ({self.muscl_limiter})", flush=True)
        print(f"  Logs directory: {self.save_dir}", flush=True)
        print("=" * 88, flush=True)

        task_specs = []
        for spec in mixtures:
            if isinstance(spec, WorkingFluid):
                p_L, p_H = paradigm_pressures(1, critical_pressure(spec))
                name, comps, mol_fracs = spec.name, spec.components, spec.mole_fractions
            else:
                # Legacy 4-tuple (name, components, mole_fractions, p_L).
                name, comps, mol_fracs, p_L = spec
                p_H = 220.0e5
            task_specs.append(
                OptimizationTask(
                    name=name,
                    components=list(comps),
                    mole_fractions=list(mol_fracs),
                    p_L=p_L,
                    p_H=p_H,
                    save_dir=str(self.save_dir),
                    N_cells=self.N_cells,
                    conv_tol=self.conv_tol,
                    use_muscl=self.use_muscl,
                    muscl_limiter=self.muscl_limiter,
                    r_init=0.50,
                    frac_init=0.40,
                    xatol=self.xatol,
                    max_iter=self.max_iter,
                )
            )

        t_start = time.time()
        print("\nRunning mixtures one at a time...", flush=True)
        results = []
        for mixture_index, spec in enumerate(task_specs, start=1):
            print(
                f"\n--- Mixture {mixture_index}/{len(task_specs)}: {spec.name} ---",
                flush=True,
            )
            result = optimize_single_mixture_worker(spec)
            results.append(result)
            print(
                f"Completed {result['name']}: eta={result['eta']:.3f}% | "
                f"W_net={result['W_net_MW']:.4f} MW | "
                f"r*={result['r_opt']:.3f} | f_rec*={result['frac_opt']:.3f}",
                flush=True,
            )

        total_time = time.time() - t_start
        print(f"\nCampaign completed in {total_time:.1f} s ({total_time/60:.2f} min).", flush=True)

        # ---------------------------------------------------------------------
        # Save CSV Summary Table
        # ---------------------------------------------------------------------
        csv_file = self.save_dir / "mixture_optima_summary.csv"
        keys = [
            "name", "x_CO2", "x_second", "T_crit", "P_crit", "P_L_bar", "P_H_bar",
            "pressure_ratio", "r_opt", "frac_opt", "r_opt_raw", "frac_opt_raw",
            "eta", "eta_best_seen", "converged_final", "max_residual_final",
            "W_net_MW", "W_turb_MW", "W_c1_MW", "W_c2_MW", "Q_in_MW",
            "htr_dT_min", "ltr_dT_min", "S_gen_HTR", "S_gen_LTR", "S_gen_Heater",
            "S_gen_Cooler", "S_gen_Turb", "S_gen_Comp1", "S_gen_Comp2",
            "S_gen_Total", "exergy_eta", "evals", "wall_time_s"
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for r in results:
                row = {k: r[k] for k in keys}
                writer.writerow(row)
        print(f"Wrote master CSV summary table to: {csv_file.name}", flush=True)

        # Print summary table
        self._print_summary_table(results)

        # Generate publication plots
        self._generate_plots(results)

        return results

    def _print_summary_table(self, results: List[Dict[str, Any]]) -> None:
        print("\n" + "=" * 96, flush=True)
        print("  OPTIMAL MIXTURE PERFORMANCE SUMMARY (Tolerance +- 0.01)", flush=True)
        print("=" * 96, flush=True)
        header = (
            f"  {'Mixture':<20} {'x_CO2':<6} {'P_L (bar)':<10} "
            f"{'r*':<6} {'f_rec*':<8} {'eta (%)':<9} {'W_net (MW)':<11} {'HTR dT':<8} {'LTR dT':<8} {'Time (s)':<8}"
        )
        print(header, flush=True)
        print("-" * 96, flush=True)
        for r in results:
            print(
                f"  {r['name']:<20} {r['x_CO2']:<6.2f} {r['P_L_bar']:<10.1f} "
                f"{r['r_opt']:<6.2f} {r['frac_opt']:<8.2f} {r['eta']:<9.3f} "
                f"{r['W_net_MW']:<11.4f} {r['htr_dT_min']:<8.1f} {r['ltr_dT_min']:<8.1f} {r['wall_time_s']:<8.1f}",
                flush=True,
            )
        print("=" * 96, flush=True)

    def _generate_plots(self, results: List[Dict[str, Any]]) -> None:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

        x_co2 = np.array([r["x_CO2"] * 100.0 for r in results])
        eta = np.array([r["eta"] for r in results])
        w_net = np.array([r["W_net_MW"] for r in results])
        r_opt = np.array([r["r_opt"] for r in results])
        frac_opt = np.array([r["frac_opt"] for r in results])
        s_htr = np.array([r["S_gen_HTR"] for r in results])
        s_ltr = np.array([r["S_gen_LTR"] for r in results])
        s_total = np.array([r["S_gen_Total"] for r in results])

        # 1. Thermal Efficiency vs CO2 Fraction
        ax1 = axes[0, 0]
        ax1.plot(x_co2, eta, "o-", color="#1f78b4", lw=2.2, markersize=8)
        ax1.set_title("Optimal Thermal Efficiency vs. Mixture Composition", fontsize=12, fontweight="bold")
        ax1.set_xlabel("CO2 Mole Fraction [%]", fontsize=11)
        ax1.set_ylabel("Optimal Thermal Efficiency $\\eta_{th}^*$ [%]", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.6)
        for x, y in zip(x_co2, eta):
            ax1.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

        # 2. Optimal Decision Variables (r* and frac_recomp*)
        ax2 = axes[0, 1]
        ax2.plot(x_co2, r_opt, "s-", color="#e31a1c", lw=2.2, markersize=8, label="Optimal Area Ratio $r^*$ ($UA_{HTR}/UA_{tot}$)")
        ax2.plot(x_co2, frac_opt, "^-", color="#2ca02c", lw=2.2, markersize=8, label="Optimal Recompression Fraction $f_{rec}^*$")
        ax2.set_title("Optimal Decision Variables vs. Mixture Composition", fontsize=12, fontweight="bold")
        ax2.set_xlabel("CO2 Mole Fraction [%]", fontsize=11)
        ax2.set_ylabel("Optimal Ratio [-]", fontsize=11)
        ax2.legend(loc="best", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.6)
        for x, y in zip(x_co2, r_opt):
            ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        for x, y in zip(x_co2, frac_opt):
            ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)

        # 3. Component Entropy Generation at Optimum
        ax3 = axes[1, 0]
        ax3.plot(x_co2, s_htr, "s-", color="#ff7f00", lw=2.0, markersize=7, label="HTR Recuperator")
        ax3.plot(x_co2, s_ltr, "d-", color="#984ea3", lw=2.0, markersize=7, label="LTR Recuperator")
        ax3.plot(x_co2, s_total, "o--", color="#377eb8", lw=2.0, markersize=7, label="Total Cycle $\\dot{S}_{gen}$")
        ax3.set_title("Component Entropy Generation at Optimum", fontsize=12, fontweight="bold")
        ax3.set_xlabel("CO2 Mole Fraction [%]", fontsize=11)
        ax3.set_ylabel("Entropy Generation Rate $\\dot{S}_{gen}$ [W/K]", fontsize=11)
        ax3.legend(loc="best", frameon=True)
        ax3.grid(True, linestyle="--", alpha=0.6)

        # 4. Net Work Output vs Composition
        ax4 = axes[1, 1]
        ax4.plot(x_co2, w_net, "v-", color="#4daf4a", lw=2.2, markersize=8)
        ax4.set_title("Net Work Output $\\dot{W}_{net}^*$ vs. Mixture Composition", fontsize=12, fontweight="bold")
        ax4.set_xlabel("CO2 Mole Fraction [%]", fontsize=11)
        ax4.set_ylabel("Net Power [MW]", fontsize=11)
        ax4.grid(True, linestyle="--", alpha=0.6)
        for x, y in zip(x_co2, w_net):
            ax4.annotate(f"{y:.3f} MW", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plot_file = self.save_dir / "mixture_optimization_results.png"
        fig.savefig(plot_file, dpi=200, bbox_inches="tight")
        print(f"Saved mixture optimization plot to: {plot_file.name}", flush=True)

        brain_dir = Path(r"C:\Users\Jaichander S\.gemini\antigravity-cli\brain\971e5329-d20b-49da-9774-2ac7ce357799")
        if brain_dir.exists():
            fig.savefig(brain_dir / "mixture_optimization_results.png", dpi=200, bbox_inches="tight")
            print("Saved artifact plot to brain directory.", flush=True)

        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Multi-Worker Mixture Optimization for Supercritical Brayton Cycle")
    parser.add_argument("--workers", type=int, default=6, help="Number of concurrent worker processes (default: 6)")
    parser.add_argument("--n-cells", type=int, default=40, help="Spatial discretization cells per stream N (default: 40)")
    parser.add_argument("--no-muscl", action="store_true", help="Disable MUSCL and use 1st-order upwind")
    parser.add_argument("--limiter", type=str, default="van_leer", choices=["van_leer", "minmod", "van_albada"])
    parser.add_argument("--tol", type=float, default=2.0, help="Residual convergence tolerance [J/(kg s)]")
    parser.add_argument("--xatol", type=float, default=0.01,
                        help="Nelder-Mead coordinate tolerance on r* and frac_recomp* (0.01 = 1%% point)")
    parser.add_argument("--max-iter", type=int, default=25, help="Nelder-Mead iteration cap")
    # --n-workers is accepted as an alias because usage.md documented it.
    parser.add_argument("--n-workers", type=int, dest="workers", help=argparse.SUPPRESS)
    args = parser.parse_args()

    opt = MixtureOptimizer(
        N_cells=args.n_cells,
        conv_tol=args.tol,
        use_muscl=not args.no_muscl,
        muscl_limiter=args.limiter,
        n_workers=args.workers,
        xatol=args.xatol,
        max_iter=args.max_iter,
    )
    opt.run_all_mixtures()


if __name__ == "__main__":
    main()
