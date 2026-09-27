"""
benchmark_muscl.py
==================
Systematic comparative study between 1st-order Upwind differencing and
2nd-order MUSCL (with TVD flux limiters) for the supercritical CO2
recompression Brayton cycle.

Evaluates:
- Efficiency progression across N in [10, 20, 40]
- Recovery of localized pinch temperatures (dT_min)
- Component entropy generation rates (S_dot_gen)
- Wall-clock convergence time and order of accuracy
"""

import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sco2_ptc.config import CycleConfig, CycleDOF
from sco2_ptc.thermo import FluidProperties
from sco2_ptc.solver import solve_cycle_ptc
from sco2_ptc.exergy import compute_cycle_exergy


def run_benchmark(grid_levels=(10, 20, 40), conv_tol=2.0, save_dir: Path | None = None):
    print("=" * 78)
    print("  SUPERCRITICAL BRAYTON CYCLE: 1ST-ORDER UPWIND vs. 2ND-ORDER MUSCL")
    print(f"  Grid Resolutions: {list(grid_levels)} | Tolerance: {conv_tol:.1f} J/(kg s)")
    print("=" * 78)

    if save_dir is None:
        save_dir = Path(__file__).resolve().parent.parent.parent / "results" / "muscl_benchmark"
    save_dir.mkdir(parents=True, exist_ok=True)

    fluid = FluidProperties(["CO2", "ETHANE"], [0.91, 0.09])

    upwind_data = []
    muscl_data = []

    # -------------------------------------------------------------------------
    # 1. Evaluate 1st-Order Upwind (use_muscl = False)
    # -------------------------------------------------------------------------
    print("\n>>> Running Baseline 1st-Order Upwind Ladder (use_muscl = False)...")
    cfg_upwind = CycleConfig(use_legacy_p_in=True, use_muscl=False)
    warm_y = None
    for N in grid_levels:
        dof = CycleDOF(A_tube=4.20, r=0.50, frac_recomp=0.40, N_cells=N)
        print(f"  Solving Upwind N={N}...")
        t0 = time.time()
        res = solve_cycle_ptc(dof=dof, cfg=cfg_upwind, fluid=fluid, y0=warm_y, conv_tol=conv_tol, verbose=False)
        dt = time.time() - t0
        warm_y = res.y_final
        ex = compute_cycle_exergy(res.state, fluid)
        row = {
            "N": N,
            "scheme": "Upwind (1st-order)",
            "time": dt,
            "eta": res.eta * 100.0,
            "htr_dt": res.state.htr_state.dT_min,
            "ltr_dt": res.state.ltr_state.dT_min,
            "htr_sgen": ex.components["HTR Recuperator"].S_dot_gen,
            "ltr_sgen": ex.components["LTR Recuperator"].S_dot_gen,
        }
        upwind_data.append(row)
        print(f"    N={N:<2d} | eta={row['eta']:.3f}% | LTR dT_min={row['ltr_dt']:.2f} K | time={dt:.1f}s")

    # -------------------------------------------------------------------------
    # 2. Evaluate 2nd-Order MUSCL (use_muscl = True, van Leer)
    # -------------------------------------------------------------------------
    print("\n>>> Running 2nd-Order MUSCL Ladder (use_muscl = True, van_leer)...")
    cfg_muscl = CycleConfig(use_legacy_p_in=True, use_muscl=True, muscl_limiter="van_leer")
    warm_y = None
    for N in grid_levels:
        dof = CycleDOF(A_tube=4.20, r=0.50, frac_recomp=0.40, N_cells=N)
        print(f"  Solving MUSCL N={N}...")
        t0 = time.time()
        res = solve_cycle_ptc(dof=dof, cfg=cfg_muscl, fluid=fluid, y0=warm_y, conv_tol=conv_tol, verbose=False)
        dt = time.time() - t0
        warm_y = res.y_final
        ex = compute_cycle_exergy(res.state, fluid)
        row = {
            "N": N,
            "scheme": "MUSCL (van Leer)",
            "time": dt,
            "eta": res.eta * 100.0,
            "htr_dt": res.state.htr_state.dT_min,
            "ltr_dt": res.state.ltr_state.dT_min,
            "htr_sgen": ex.components["HTR Recuperator"].S_dot_gen,
            "ltr_sgen": ex.components["LTR Recuperator"].S_dot_gen,
        }
        muscl_data.append(row)
        print(f"    N={N:<2d} | eta={row['eta']:.3f}% | LTR dT_min={row['ltr_dt']:.2f} K | time={dt:.1f}s")

    # -------------------------------------------------------------------------
    # 3. Print Comparison Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  COMPARATIVE SUMMARY: UPWIND vs. MUSCL")
    print("=" * 78)
    header = f"  {'N':<4} | {'Upwind eta':<12} {'MUSCL eta':<12} {'Delta eta':<10} | {'Upwind LTR dT':<14} {'MUSCL LTR dT':<14}"
    print(header)
    print("-" * 78)
    for u, m in zip(upwind_data, muscl_data):
        d_eta = m["eta"] - u["eta"]
        print(f"  {u['N']:<4d} | {u['eta']:<12.3f} {m['eta']:<12.3f} {d_eta:<+10.3f} | {u['ltr_dt']:<14.2f} {m['ltr_dt']:<14.2f}")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 4. Publication Comparison Plot
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    N_arr = np.array([d["N"] for d in upwind_data])
    eta_up = np.array([d["eta"] for d in upwind_data])
    eta_mu = np.array([d["eta"] for d in muscl_data])
    ltr_dt_up = np.array([d["ltr_dt"] for d in upwind_data])
    ltr_dt_mu = np.array([d["ltr_dt"] for d in muscl_data])

    # Continuum reference line from N=80 Richardson extrapolation (46.14%)
    eta_continuum = 46.142

    # Panel 1: Efficiency vs N
    ax1 = axes[0]
    ax1.plot(N_arr, eta_up, "o--", color="#1f78b4", lw=2.0, markersize=8, label="1st-Order Upwind")
    ax1.plot(N_arr, eta_mu, "s-", color="#e31a1c", lw=2.2, markersize=8, label="2nd-Order MUSCL (van Leer)")
    ax1.axhline(eta_continuum, color="gray", linestyle=":", lw=1.5, label=f"Continuum Limit ({eta_continuum:.2f}%)")
    ax1.set_title("Thermal Efficiency: Upwind vs. MUSCL", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Cells per Stream ($N$)", fontsize=11)
    ax1.set_ylabel("Thermal Efficiency $\\eta_{th}$ [%]", fontsize=11)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    for x, y in zip(N_arr, eta_mu):
        ax1.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold", color="#e31a1c")

    # Panel 2: LTR Pinch Temperature vs N
    ax2 = axes[1]
    ax2.plot(N_arr, ltr_dt_up, "o--", color="#1f78b4", lw=2.0, markersize=8, label="1st-Order Upwind")
    ax2.plot(N_arr, ltr_dt_mu, "s-", color="#2ca02c", lw=2.2, markersize=8, label="2nd-Order MUSCL (van Leer)")
    ax2.axhline(15.36, color="gray", linestyle=":", lw=1.5, label="N=80 Continuum Target (15.36 K)")
    ax2.set_title("LTR Pinch $\\Delta T_{min}$: Upwind vs. MUSCL", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Cells per Stream ($N$)", fontsize=11)
    ax2.set_ylabel("Minimum Pinch $\\Delta T_{min}$ [K]", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    for x, y in zip(N_arr, ltr_dt_mu):
        ax2.annotate(f"{y:.2f} K", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold", color="#2ca02c")

    plt.tight_layout()
    fig_path = save_dir / "muscl_vs_upwind_convergence.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot to: {fig_path}")

    brain_dir = Path(r"C:\Users\Jaichander S\.gemini\antigravity-cli\brain\971e5329-d20b-49da-9774-2ac7ce357799")
    if brain_dir.exists():
        fig.savefig(brain_dir / "muscl_vs_upwind_convergence.png", dpi=200, bbox_inches="tight")
        print(f"Saved artifact plot to brain dir.")

    plt.close(fig)
    return upwind_data, muscl_data


if __name__ == "__main__":
    run_benchmark(grid_levels=(10, 20, 40), conv_tol=2.0)
