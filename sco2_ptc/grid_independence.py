"""
grid_independence.py
====================
Systematic Grid Independence (Mesh Convergence) Study for the Supercritical CO2
Recompression Brayton Cycle. Evaluates discretization levels N in [10, 20, 40, 80]
using progressive warm-started laddering to conv_tol = 2.0 J/(kg s).

Quantifies asymptotic convergence, Richardson extrapolation, and discretization
errors for:
1. First-Law Thermal Efficiency (eta)
2. Recuperator Duties and Pinch Temperatures (dT_min)
3. Component Entropy Generation Rates (S_dot_gen_HTR, S_dot_gen_LTR, S_dot_gen_total)

Saves all converged solutions as .npz archives and generates publication-grade plots.
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure parent python directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sco2_ptc.config import CycleConfig, CycleDOF
from sco2_ptc.thermo import FluidProperties
from sco2_ptc.solver import solve_cycle_ptc, CycleResult
from sco2_ptc.exergy import compute_cycle_exergy, CycleExergyAnalysis
from sco2_ptc.io import save_cycle_result


def run_grid_independence(
    grid_levels=(10, 20, 40, 80),
    conv_tol: float = 2.0,
    save_dir: Path | None = None,
):
    print("=" * 78)
    print("  SUPERCRITICAL BRAYTON CYCLE: GRID INDEPENDENCE STUDY")
    print(f"  Discretization Levels: {list(grid_levels)} | Tolerance: {conv_tol:.1f} J/(kg s)")
    print("=" * 78)

    if save_dir is None:
        save_dir = Path(__file__).resolve().parent.parent.parent / "results" / "grid_independence"
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = CycleConfig(use_legacy_p_in=True)
    fluid = FluidProperties(cfg.fluid_components, cfg.mole_fractions)

    results_dict = {}
    exergy_dict = {}
    warm_y = None

    t_total_start = time.time()

    for idx, N in enumerate(grid_levels):
        dof = CycleDOF(A_tube=4.20, r=0.50, frac_recomp=0.40, N_cells=N)
        print(f"\n>>> Solving Grid Level {idx + 1}/{len(grid_levels)}: N={N} (states={4 * N + 1})...")

        t_stage = time.time()
        res = solve_cycle_ptc(
            dof=dof,
            cfg=cfg,
            fluid=fluid,
            y0=warm_y,
            tau_max=1000.0,
            chunk_dtau=25.0,
            conv_tol=conv_tol,
            verbose=True,
        )
        stage_time = time.time() - t_stage
        warm_y = res.y_final

        # Compute post-convergence Second-Law exergy analysis
        exergy = compute_cycle_exergy(res.state, fluid, T0=298.15)

        # Save to disk
        archive_path = save_dir / f"result_N{N}.npz"
        save_cycle_result(res, archive_path, exergy=exergy)
        print(f"  Saved converged result to: {archive_path.name} ({stage_time:.2f} s)")

        results_dict[N] = res
        exergy_dict[N] = exergy

    total_study_time = time.time() - t_total_start

    # -------------------------------------------------------------------------
    # Tabular Metrics Compilation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  GRID INDEPENDENCE RESULTS SUMMARY")
    print("=" * 78)
    header = f"  {'N':<5} {'States':<7} {'Time (s)':<9} {'eta (%)':<10} {'W_net (MW)':<11} {'HTR dT_min':<11} {'LTR dT_min':<11} {'S_gen_HTR':<11} {'S_gen_LTR':<11}"
    print(header)
    print("-" * 78)

    table_data = []
    for N in grid_levels:
        r = results_dict[N]
        e = exergy_dict[N]
        s_htr = e.components["HTR Recuperator"].S_dot_gen
        s_ltr = e.components["LTR Recuperator"].S_dot_gen
        row = {
            "N": N,
            "states": 4 * N + 1,
            "time": r.wall_time_s,
            "eta": r.eta * 100.0,
            "W_net": r.W_dot_net / 1e6,
            "htr_dT_min": r.state.htr_state.dT_min,
            "ltr_dT_min": r.state.ltr_state.dT_min,
            "s_htr": s_htr,
            "s_ltr": s_ltr,
            "s_total": e.S_dot_gen_total,
        }
        table_data.append(row)
        print(
            f"  {N:<5d} {row['states']:<7d} {row['time']:<9.1f} {row['eta']:<10.4f} "
            f"{row['W_net']:<11.5f} {row['htr_dT_min']:<11.2f} {row['ltr_dT_min']:<11.2f} "
            f"{row['s_htr']:<11.2f} {row['s_ltr']:<11.2f}"
        )

    # -------------------------------------------------------------------------
    # Relative Errors Between Consecutive Grids
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("  CONVERGENCE DELTAS BETWEEN CONSECUTIVE GRIDS:")
    print("-" * 78)
    print(f"  {'Refinement Step':<20} {'|Delta eta| (%)':<18} {'|Delta S_HTR| (%)':<18} {'|Delta S_LTR| (%)':<18}")
    print("-" * 78)
    for i in range(1, len(table_data)):
        prev = table_data[i - 1]
        curr = table_data[i]
        d_eta = abs(curr["eta"] - prev["eta"]) / curr["eta"] * 100.0
        d_s_htr = abs(curr["s_htr"] - prev["s_htr"]) / curr["s_htr"] * 100.0
        d_s_ltr = abs(curr["s_ltr"] - prev["s_ltr"]) / curr["s_ltr"] * 100.0
        step_str = f"N={prev['N']} -> N={curr['N']}"
        print(f"  {step_str:<20} {d_eta:<18.4f} {d_s_htr:<18.4f} {d_s_ltr:<18.4f}")

    # Richardson extrapolation for N = [20, 40, 80]
    if len(grid_levels) >= 3 and 20 in results_dict and 40 in results_dict and 80 in results_dict:
        f20 = results_dict[20].eta * 100.0
        f40 = results_dict[40].eta * 100.0
        f80 = results_dict[80].eta * 100.0
        diff1 = f40 - f20
        diff2 = f80 - f40
        if diff1 * diff2 > 0:
            order_p = np.log(abs(diff1 / diff2)) / np.log(2.0)
            f_extrap = f80 + (f80 - f40) / (2.0**order_p - 1.0)
            print("-" * 78)
            print(f"  Richardson Extrapolation (N=20, 40, 80):")
            print(f"  Estimated Spatial Order of Convergence p : {order_p:.2f}")
            print(f"  Extrapolated Continuum Efficiency eta_h=0: {f_extrap:.4f} %")
            print(f"  N=80 Discretization Error vs Continuum   : {abs(f80 - f_extrap):.4f} % ({(abs(f80 - f_extrap)/f_extrap)*100:.3f}%)")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Publication-Grade 4-Panel Grid Independence Plot
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    N_arr = np.array([d["N"] for d in table_data])
    eta_arr = np.array([d["eta"] for d in table_data])
    s_htr_arr = np.array([d["s_htr"] for d in table_data])
    s_ltr_arr = np.array([d["s_ltr"] for d in table_data])
    htr_dt_arr = np.array([d["htr_dT_min"] for d in table_data])
    ltr_dt_arr = np.array([d["ltr_dT_min"] for d in table_data])

    # 1. Thermal Efficiency vs N
    ax1 = axes[0, 0]
    ax1.plot(N_arr, eta_arr, "o-", color="#1f78b4", lw=2.2, markersize=8, label="$\\eta_{th}$ (Simulated)")
    ax1.set_title("Thermal Efficiency vs. Discretization Resolution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Cells per HX Stream ($N$)", fontsize=11)
    ax1.set_ylabel("Thermal Efficiency $\\eta_{th}$ [%]", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    for x, y in zip(N_arr, eta_arr):
        ax1.annotate(f"{y:.3f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    # 2. Entropy Generation Rates vs N
    ax2 = axes[0, 1]
    ax2.plot(N_arr, s_htr_arr, "s-", color="#d95f02", lw=2.2, markersize=8, label="HTR $\\dot{S}_{gen}$")
    ax2.plot(N_arr, s_ltr_arr, "^-", color="#2ca02c", lw=2.2, markersize=8, label="LTR $\\dot{S}_{gen}$")
    ax2.set_title("Recuperator Entropy Generation vs. Discretization", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Cells per HX Stream ($N$)", fontsize=11)
    ax2.set_ylabel("Entropy Generation Rate $\\dot{S}_{gen}$ [W/K]", fontsize=11)
    ax2.legend(loc="best", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Minimum Pinch Temperatures vs N
    ax3 = axes[1, 0]
    ax3.plot(N_arr, htr_dt_arr, "s-", color="#7570b3", lw=2.2, markersize=8, label="HTR Pinch $\\Delta T_{min}$")
    ax3.plot(N_arr, ltr_dt_arr, "d-", color="#e7298a", lw=2.2, markersize=8, label="LTR Pinch $\\Delta T_{min}$")
    ax3.set_title("Minimum Pinch Temperature Difference vs. $N$", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Number of Cells per HX Stream ($N$)", fontsize=11)
    ax3.set_ylabel("Pinch $\\Delta T_{min}$ [K]", fontsize=11)
    ax3.legend(loc="best", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # 4. Relative Discretization Error vs 1/N (Convergence Rate)
    ax4 = axes[1, 1]
    inv_N = 1.0 / N_arr
    # Error relative to finest N=80
    ref_eta = eta_arr[-1]
    rel_err_eta = np.abs(eta_arr - ref_eta) / ref_eta * 100.0
    # Avoid log of 0 for last point
    valid_mask = rel_err_eta > 1e-6
    if np.any(valid_mask):
        ax4.loglog(inv_N[valid_mask], rel_err_eta[valid_mask], "o-", color="#e41a1c", lw=2.2, markersize=8, label="Relative Error vs $N=80$")
        # 1st order reference line
        ref_x = inv_N[valid_mask]
        ref_y = ref_x * (rel_err_eta[valid_mask][0] / ref_x[0])
        ax4.loglog(ref_x, ref_y, "--", color="gray", lw=1.5, label="$\mathcal{O}(\\Delta x)$ Reference")
    ax4.set_title("Spatial Discretization Error Scaling", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Reciprocal Grid Dimension ($1/N$)", fontsize=11)
    ax4.set_ylabel("Relative Error in Efficiency [%]", fontsize=11)
    ax4.legend(loc="best", frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    fig_file = save_dir / "grid_independence_study.png"
    fig.savefig(fig_file, dpi=200, bbox_inches="tight")
    print(f"\nSaved Grid Independence plot to: {fig_file}")

    # Also save to conversation artifact directory
    brain_dir = Path(r"C:\Users\Jaichander S\.gemini\antigravity-cli\brain\971e5329-d20b-49da-9774-2ac7ce357799")
    if brain_dir.exists():
        brain_fig = brain_dir / "grid_independence_study.png"
        fig.savefig(brain_fig, dpi=200, bbox_inches="tight")
        print(f"Saved artifact plot to: {brain_fig}")

    plt.close(fig)
    return table_data


if __name__ == "__main__":
    run_grid_independence(grid_levels=(10, 20, 40, 80), conv_tol=2.0)
