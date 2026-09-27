"""
run_three_paradigms.py
======================
Autonomous multi-worker campaign executing all 3 thermodynamic comparison paradigms
across 6 fluid compositions (Pure sCO2, 95%, 90%, 85%, 80%, 75% CO2):

Paradigm 1: Fixed Pressures ("Hardware Fair / Drop-in")
            P_L = 83.8 bar, P_H = 220.0 bar, Pi = 2.625 (strictly fixed)

Paradigm 2: Critical Proximity with Fixed P_H ("Thermodynamic Sweet Spot")
            P_L = P_c + 10 bar, P_H = 220.0 bar, Pi = 2.63 -> 3.04

Paradigm 3: Normalized Pressure Ratio ("Thermodynamic & Expansion Fair")
            P_L = P_c + 10 bar, P_H = 2.6253 * P_L, Pi = 2.6253 (strictly fixed)

Features:
- Parallel execution across 6 worker processes using ProcessPoolExecutor.
- Nelder-Mead optimization to +-0.01 (1% pt) tolerance on r* and frac_recomp*.
- Continuous warm-starting with PTC solver and 2nd-order MUSCL discretization.
- Second-Law component-wise exergy destruction breakdown.
- Generates master comparison CSVs and high-resolution publication plots.
- Complete unbuffered logging to disk.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
import time
import csv
import argparse
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt

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

# Ensure package is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sco2_ptc.mixture_optimization import OptimizationTask, optimize_single_mixture_worker


# Fluid specifications: (Name, components, mole_fractions, P_crit_bar approx)
BASE_FLUIDS = [
    ("Pure sCO2", ["CO2"], [1.00], 73.77),
    ("95% CO2 + 5% C2H6", ["CO2", "ETHANE"], [0.95, 0.05], 71.72),
    ("90% CO2 + 10% C2H6", ["CO2", "ETHANE"], [0.90, 0.10], 69.24),
    ("85% CO2 + 15% C2H6", ["CO2", "ETHANE"], [0.85, 0.15], 66.70),
    ("80% CO2 + 20% C2H6", ["CO2", "ETHANE"], [0.80, 0.20], 64.35),
    ("75% CO2 + 25% C2H6", ["CO2", "ETHANE"], [0.75, 0.25], 62.31),
]

PI_BASELINE = 220.0 / 83.8  # 2.625298


def build_task_specs_for_paradigm(
    paradigm_id: int,
    out_dir: Path,
    N_cells: int = 40,
    conv_tol: float = 2.0,
    use_muscl: bool = True,
    muscl_limiter: str = "van_leer",
    xatol: float = 0.01,
) -> List[OptimizationTask]:
    """Generates task specifications for a given comparison paradigm."""
    specs = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, comps, fracs, pc_bar in BASE_FLUIDS:
        if paradigm_id == 1:
            # Paradigm 1: Fixed Pressures
            p_L = 83.8e5
            p_H = 220.0e5
        elif paradigm_id == 2:
            # Paradigm 2: Critical Proximity, Fixed P_H
            p_L = (pc_bar + 10.0) * 1e5
            p_H = 220.0e5
        elif paradigm_id == 3:
            # Paradigm 3: Normalized Pressure Ratio
            p_L = (pc_bar + 10.0) * 1e5
            p_H = PI_BASELINE * p_L
        else:
            raise ValueError(f"Unknown paradigm_id: {paradigm_id}")

        spec = OptimizationTask(
            name=name,
            components=comps,
            mole_fractions=fracs,
            p_L=p_L,
            p_H=p_H,
            save_dir=str(out_dir),
            N_cells=N_cells,
            conv_tol=conv_tol,
            use_muscl=use_muscl,
            muscl_limiter=muscl_limiter,
            r_init=0.50,
            frac_init=0.35,
            r_bounds=(0.25, 0.75),
            frac_bounds=(0.15, 0.55),
            xatol=xatol,
            max_iter=25,
        )
        specs.append(spec)

    return specs


def run_single_paradigm(
    paradigm_id: int,
    title: str,
    save_dir: Path,
    workers: int = 6,
    N_cells: int = 40,
    conv_tol: float = 2.0,
    use_muscl: bool = True,
    muscl_limiter: str = "van_leer",
    xatol: float = 0.01,
) -> List[Dict[str, Any]]:
    """Runs all 6 mixtures for one comparison paradigm in parallel."""
    print("\n" + "=" * 92, flush=True)
    print(f"  LAUNCHING PARADIGM {paradigm_id}: {title}", flush=True)
    print(f"  Workers: {workers} | Grid: N={N_cells} | MUSCL: {use_muscl} ({muscl_limiter}) | xatol: {xatol}", flush=True)
    print(f"  Directory: {save_dir}", flush=True)
    print("=" * 92, flush=True)

    # Check if paradigm was already completed
    csv_file = save_dir / f"paradigm_{paradigm_id}_summary.csv"
    if csv_file.exists():
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                if len(reader) == len(BASE_FLUIDS):
                    print(f"  [Resume] Paradigm {paradigm_id} already fully completed! Loaded {len(reader)} fluids from {csv_file.name}.", flush=True)
                    results = []
                    for row in reader:
                        r_typed = {}
                        for k, v in row.items():
                            try:
                                r_typed[k] = float(v)
                            except ValueError:
                                r_typed[k] = v
                        results.append(r_typed)
                    return results
        except Exception as e_csv:
            print(f"  Notice: Could not parse existing summary CSV ({e_csv}), executing paradigm.", flush=True)

    task_specs = build_task_specs_for_paradigm(
        paradigm_id=paradigm_id,
        out_dir=save_dir,
        N_cells=N_cells,
        conv_tol=conv_tol,
        use_muscl=use_muscl,
        muscl_limiter=muscl_limiter,
        xatol=xatol,
    )

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(workers, len(task_specs))) as executor:
        results = list(executor.map(optimize_single_mixture_worker, task_specs))

    elapsed = time.time() - t0
    print(f"Paradigm {paradigm_id} completed in {elapsed:.1f} s ({elapsed/60:.2f} min).", flush=True)

    # Save summary CSV
    csv_file = save_dir / f"paradigm_{paradigm_id}_summary.csv"
    keys = list(results[0].keys())
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Saved summary CSV: {csv_file}", flush=True)

    return results


def generate_master_plots_and_tables(
    results_p1: List[Dict[str, Any]],
    results_p2: List[Dict[str, Any]],
    results_p3: List[Dict[str, Any]],
    output_dir: Path,
):
    """Generates cross-paradigm publication figures and master comparison tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Master CSV
    master_csv = output_dir / "master_three_paradigms_summary.csv"
    keys = ["paradigm", "name", "x_CO2", "P_L_bar", "P_H_bar", "pressure_ratio", "r_opt", "frac_opt", "eta", "W_net_MW", "S_gen_Total", "exergy_eta"]
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for p_name, r_list in [
            ("Fixed_Pressures", results_p1),
            ("Critical_Proximity_Fixed_PH", results_p2),
            ("Normalized_Pressure_Ratio", results_p3),
        ]:
            for r in r_list:
                row = {
                    "paradigm": p_name,
                    "name": r["name"],
                    "x_CO2": r["x_CO2"],
                    "P_L_bar": r["P_L_bar"],
                    "P_H_bar": r["P_H_bar"],
                    "pressure_ratio": r["pressure_ratio"],
                    "r_opt": r["r_opt"],
                    "frac_opt": r["frac_opt"],
                    "eta": r["eta"],
                    "W_net_MW": r["W_net_MW"],
                    "S_gen_Total": r["S_gen_Total"],
                    "exergy_eta": r["exergy_eta"],
                }
                writer.writerow(row)
    print(f"Master comparison CSV saved to: {master_csv}", flush=True)

    # 2. Cross-Comparison Publication Plot (4 Subplots)
    x1 = [r["x_CO2"] * 100 for r in results_p1]
    eta1 = [r["eta"] for r in results_p1]
    r1 = [r["r_opt"] for r in results_p1]
    f1 = [r["frac_opt"] for r in results_p1]
    s1 = [r["S_gen_Total"] / 1e3 for r in results_p1]

    x2 = [r["x_CO2"] * 100 for r in results_p2]
    eta2 = [r["eta"] for r in results_p2]
    r2 = [r["r_opt"] for r in results_p2]
    f2 = [r["frac_opt"] for r in results_p2]
    s2 = [r["S_gen_Total"] / 1e3 for r in results_p2]

    x3 = [r["x_CO2"] * 100 for r in results_p3]
    eta3 = [r["eta"] for r in results_p3]
    r3 = [r["r_opt"] for r in results_p3]
    f3 = [r["frac_opt"] for r in results_p3]
    s3 = [r["S_gen_Total"] / 1e3 for r in results_p3]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Thermal Efficiency
    ax = axes[0, 0]
    ax.plot(x1, eta1, "o-", color="#1f77b4", lw=2.2, label="Paradigm 1: Fixed Pressures (P_L=83.8 bar, P_H=220 bar)")
    ax.plot(x2, eta2, "s--", color="#d62728", lw=2.2, label="Paradigm 2: Critical Proximity (P_L=Pc+10 bar, P_H=220 bar)")
    ax.plot(x3, eta3, "^-.", color="#2ca02c", lw=2.2, label="Paradigm 3: Normalized Ratio (P_L=Pc+10 bar, Pi=2.625)")
    ax.set_xlabel("CO2 Mole Fraction (%)", fontsize=11)
    ax.set_ylabel("Optimal Cycle Thermal Efficiency (%)", fontsize=11)
    ax.set_title("(a) Thermal Efficiency Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")

    # (b) Optimal Area Allocation Ratio r*
    ax = axes[0, 1]
    ax.plot(x1, r1, "o-", color="#1f77b4", lw=2.2, label="Paradigm 1: Fixed Pressures")
    ax.plot(x2, r2, "s--", color="#d62728", lw=2.2, label="Paradigm 2: Critical Proximity")
    ax.plot(x3, r3, "^-.", color="#2ca02c", lw=2.2, label="Paradigm 3: Normalized Ratio")
    ax.set_xlabel("CO2 Mole Fraction (%)", fontsize=11)
    ax.set_ylabel("Optimal Area Allocation Ratio r* [-]", fontsize=11)
    ax.set_title("(b) Optimal Recuperator Area Fraction (r* = A_LTR / A_tot)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # (c) Optimal Recompression Ratio frac_recomp*
    ax = axes[1, 0]
    ax.plot(x1, f1, "o-", color="#1f77b4", lw=2.2, label="Paradigm 1: Fixed Pressures")
    ax.plot(x2, f2, "s--", color="#d62728", lw=2.2, label="Paradigm 2: Critical Proximity")
    ax.plot(x3, f3, "^-.", color="#2ca02c", lw=2.2, label="Paradigm 3: Normalized Ratio")
    ax.set_xlabel("CO2 Mole Fraction (%)", fontsize=11)
    ax.set_ylabel("Optimal Recompression Fraction f_recomp* [-]", fontsize=11)
    ax.set_title("(c) Optimal Recompression Flow Split Ratio", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # (d) Total Entropy Generation Rate
    ax = axes[1, 1]
    ax.plot(x1, s1, "o-", color="#1f77b4", lw=2.2, label="Paradigm 1: Fixed Pressures")
    ax.plot(x2, s2, "s--", color="#d62728", lw=2.2, label="Paradigm 2: Critical Proximity")
    ax.plot(x3, s3, "^-.", color="#2ca02c", lw=2.2, label="Paradigm 3: Normalized Ratio")
    ax.set_xlabel("CO2 Mole Fraction (%)", fontsize=11)
    ax.set_ylabel("Total Cycle Entropy Generation (kW/K)", fontsize=11)
    ax.set_title("(d) Total Exergy Destruction Rate", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plot_file = output_dir / "three_paradigms_comparison.png"
    plt.savefig(plot_file, dpi=250)
    plt.close()
    print(f"Master comparative plot saved to: {plot_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Autonomous Three-Paradigm Optimization Campaign")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent worker processes")
    parser.add_argument("--n-cells", type=int, default=40, help="Grid cells per stream")
    parser.add_argument("--xatol", type=float, default=0.01, help="Tolerance for r* and f_recomp* (0.01 = 1%% pt)")
    parser.add_argument("--no-muscl", action="store_true", help="Disable MUSCL and revert to 1st-order upwind")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    campaign_dir = repo_root / "results" / "three_paradigms_campaign"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    use_muscl = not args.no_muscl
    total_campaign_start = time.time()

    print("==========================================================================================", flush=True)
    print("       STARTING AUTONOMOUS OVERNIGHT OPTIMIZATION CAMPAIGN ACROSS ALL 3 PARADIGMS        ", flush=True)
    print(f"  Target Compositions : 6 fluids (Pure, 95%, 90%, 85%, 80%, 75% CO2)", flush=True)
    print(f"  Grid Resolution     : N = {args.n_cells} cells/stream", flush=True)
    print(f"  Advection Scheme    : {'2nd-Order MUSCL (van Leer)' if use_muscl else '1st-Order Upwind'}", flush=True)
    print(f"  Nelder-Mead Tol     : xatol = {args.xatol} (1% point precision)", flush=True)
    print(f"  Parallel Workers    : {args.workers} CPU cores", flush=True)
    print(f"  Campaign Directory  : {campaign_dir}", flush=True)
    print("==========================================================================================", flush=True)

    # -------------------------------------------------------------------------
    # Study 1: Fixed Pressures
    # -------------------------------------------------------------------------
    dir_p1 = campaign_dir / "study_1_fixed_pressures"
    res_p1 = run_single_paradigm(
        paradigm_id=1,
        title="Fixed Pressures (P_L = 83.8 bar, P_H = 220.0 bar, Pi = 2.625)",
        save_dir=dir_p1,
        workers=args.workers,
        N_cells=args.n_cells,
        use_muscl=use_muscl,
        xatol=args.xatol,
    )

    # -------------------------------------------------------------------------
    # Study 2: Critical Proximity, Fixed P_H = 220 bar
    # -------------------------------------------------------------------------
    dir_p2 = campaign_dir / "study_2_critical_proximity"
    res_p2 = run_single_paradigm(
        paradigm_id=2,
        title="Critical Proximity (P_L = Pc + 10 bar, P_H = 220.0 bar, Variable Pi)",
        save_dir=dir_p2,
        workers=args.workers,
        N_cells=args.n_cells,
        use_muscl=use_muscl,
        xatol=args.xatol,
    )

    # -------------------------------------------------------------------------
    # Study 3: Normalized Pressure Ratio
    # -------------------------------------------------------------------------
    dir_p3 = campaign_dir / "study_3_normalized_ratio"
    res_p3 = run_single_paradigm(
        paradigm_id=3,
        title="Normalized Pressure Ratio (P_L = Pc + 10 bar, Pi = 2.625 Constant)",
        save_dir=dir_p3,
        workers=args.workers,
        N_cells=args.n_cells,
        use_muscl=use_muscl,
        xatol=args.xatol,
    )

    # -------------------------------------------------------------------------
    # Compile Master Summary, Plots, and Artifact
    # -------------------------------------------------------------------------
    print("\nCompiling master cross-paradigm comparisons and generating publication figures...", flush=True)
    generate_master_plots_and_tables(res_p1, res_p2, res_p3, campaign_dir)

    total_time = time.time() - total_campaign_start
    print("\n" + "=" * 92, flush=True)
    print(f"  ALL 3 PARADIGMS COMPLETED SUCCESSFULLY IN {total_time:.1f} s ({total_time/60:.2f} min)!", flush=True)
    print(f"  All logs, CSV summaries, and high-res plots are saved in: {campaign_dir}", flush=True)
    print("=" * 92, flush=True)


if __name__ == "__main__":
    main()
