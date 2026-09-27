"""
benchmark_case_c.py
===================
Executes Case C (Calibrated Adaptive Tolerances: rtol <= 5e-4, atol <= 2e2)
and benchmarks performance against Case A (Fixed rtol=1e-4) and Case B (Uncalibrated rtol=1e-2).
"""

import sys
from pathlib import Path
import time
import numpy as np

# Ensure parent python directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sco2_ptc.config import CycleConfig, CycleDOF
from sco2_ptc.thermo import FluidProperties
from sco2_ptc.solver import solve_cycle_ladder


def run_case_c():
    print("=" * 76)
    print("  CASE C BENCHMARK: Calibrated Adaptive Tolerances (rtol <= 5e-4)")
    print("=" * 76)

    cfg = CycleConfig()
    fluid = FluidProperties(cfg.fluid_components, cfg.mole_fractions)
    target_dof = CycleDOF(A_tube=4.20, r=0.50, frac_recomp=0.40, N_cells=40)
    ladder = [10, 20, 40]

    # Baseline recorded results from identical environment:
    case_a = {
        "name": "Case A (Fixed)",
        "desc": "rtol=1e-4, atol=1e2",
        "total_time": 440.17,
        "stage3_time": 355.04,
        "n_fev": 916,
        "n_chunks": 14,
        "max_res": 1.317e0,
        "eta": 45.0722,
        "W_net": 1.47341,
        "W_turb": 2.05526,
        "T_turb_out": 733.3981,
        "T_mix_out": 467.4150,
        "eb_err": 809.11,
    }

    case_b = {
        "name": "Case B (Uncalibrated)",
        "desc": "rtol=1e-2, atol=1e3",
        "total_time": 682.92,
        "stage3_time": 521.42,
        "n_fev": 1233,
        "n_chunks": 14,
        "max_res": 1.437e0,
        "eta": 45.0611,
        "W_net": 1.47360,
        "W_turb": 2.05526,
        "T_turb_out": 733.3981,
        "T_mix_out": 467.2213,
        "eb_err": 808.92,
    }

    print("\n>>> Running CASE C: Calibrated Adaptive Tolerances (rtol <= 5e-4)...")
    fluid.clear_cell_cache()
    t0_c = time.time()
    res_c = solve_cycle_ladder(
        target_dof=target_dof,
        cfg=cfg,
        fluid=fluid,
        ladder=ladder,
        conv_tol=2.0,
        adaptive_tol=True,
        verbose=True,
    )
    time_c = time.time() - t0_c

    # Comparison Table
    print("\n" + "=" * 76)
    print("  A / B / C COMPREHENSIVE BENCHMARK COMPARISON (N=10 -> 20 -> 40, tol=2.0)")
    print("=" * 76)
    print(f"  {'Metric':<26} {'Case A (Fixed)':<16} {'Case B (Loose)':<16} {'Case C (Calibrated)':<16}")
    print(f"  {'Configuration':<26} {'rtol=1e-4':<16} {'rtol=1e-2':<16} {'rtol<=5e-4':<16}")
    print("-" * 76)
    print(f"  {'Total Wall Time (s)':<26} {case_a['total_time']:<16.2f} {case_b['total_time']:<16.2f} {time_c:<16.2f}")
    speedup_vs_a = case_a['total_time'] / time_c if time_c > 0 else 0.0
    speedup_vs_b = case_b['total_time'] / time_c if time_c > 0 else 0.0
    print(f"  {'Speedup vs Case A':<26} {'1.00x (baseline)':<16} {'0.64x (slowdown)':<16} {speedup_vs_a:.2f}x")
    print(f"  {'Final Stage Time (s)':<26} {case_a['stage3_time']:<16.2f} {case_b['stage3_time']:<16.2f} {res_c.wall_time_s:<16.2f}")
    print(f"  {'Final Stage ODE fev':<26} {case_a['n_fev']:<16d} {case_b['n_fev']:<16d} {res_c.n_fev:<16d}")
    print(f"  {'Final Stage Chunks':<26} {case_a['n_chunks']:<16d} {case_b['n_chunks']:<16d} {res_c.n_chunks:<16d}")
    print(f"  {'Final Residual (J/kg-s)':<26} {case_a['max_res']:<16.3e} {case_b['max_res']:<16.3e} {res_c.max_residual:<16.3e}")
    print("-" * 76)
    print("  THERMODYNAMIC PARITY VERIFICATION:")
    print(f"  {'Thermal Efficiency (%)':<26} {case_a['eta']:<16.4f} {case_b['eta']:<16.4f} {res_c.eta * 100:<16.4f}")
    print(f"  {'Net Power (MW)':<26} {case_a['W_net']:<16.5f} {case_b['W_net']:<16.5f} {res_c.W_dot_net / 1e6:<16.5f}")
    print(f"  {'Turbine Power (MW)':<26} {case_a['W_turb']:<16.5f} {case_b['W_turb']:<16.5f} {res_c.W_dot_turb / 1e6:<16.5f}")
    print(f"  {'Turbine Outlet T (K)':<26} {case_a['T_turb_out']:<16.4f} {case_b['T_turb_out']:<16.4f} {res_c.state.stations[6]['T']:<16.4f}")
    print(f"  {'Mixer Outlet T (K)':<26} {case_a['T_mix_out']:<16.4f} {case_b['T_mix_out']:<16.4f} {res_c.state.stations[12]['T']:<16.4f}")
    print(f"  {'Energy Balance Err (W)':<26} {case_a['eb_err']:<16.2f} {case_b['eb_err']:<16.2f} {res_c.state.energy_balance_err:<16.2f}")
    print("=" * 76)


if __name__ == "__main__":
    run_case_c()
