"""
benchmark_ab_tol.py
===================
A/B testing script to measure the speedup and numerical consistency of
Adaptive Newton / BDF error tolerances during pseudotransient continuation.

Case A: Fixed tight tolerances (rtol=1e-4, atol=1e2)
Case B: Adaptive tolerances (loose during transition, tightening at terminal approach)
"""

import sys
from pathlib import Path
import time
import numpy as np

# Ensure parent python directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sco2_ptc.config import CycleConfig, CycleDOF
from sco2_ptc.thermo import FluidProperties
from sco2_ptc.solver import solve_cycle_ladder, solve_cycle_ptc


def run_ab_test():
    print("=" * 72)
    print("  A/B BENCHMARK: Fixed vs. Adaptive Newton Tolerances in PTC")
    print("=" * 72)

    cfg = CycleConfig()
    fluid = FluidProperties(cfg.fluid_components, cfg.mole_fractions)
    target_dof = CycleDOF(A_tube=4.20, r=0.50, frac_recomp=0.40, N_cells=40)
    ladder = [10, 20, 40]

    # -------------------------------------------------------------------------
    # CASE A: Fixed Tolerances
    # -------------------------------------------------------------------------
    print("\n>>> Running CASE A: Fixed Tolerances (rtol=1e-4, atol=1e2)...")
    fluid.clear_cell_cache()
    t0_a = time.time()
    res_a = solve_cycle_ladder(
        target_dof=target_dof,
        cfg=cfg,
        fluid=fluid,
        ladder=ladder,
        conv_tol=2.0,
        adaptive_tol=False,
        verbose=True,
    )
    time_a = time.time() - t0_a

    # -------------------------------------------------------------------------
    # CASE B: Adaptive Tolerances
    # -------------------------------------------------------------------------
    print("\n>>> Running CASE B: Adaptive Tolerances (dynamic rtol/atol)...")
    fluid.clear_cell_cache()
    t0_b = time.time()
    res_b = solve_cycle_ladder(
        target_dof=target_dof,
        cfg=cfg,
        fluid=fluid,
        ladder=ladder,
        conv_tol=2.0,
        adaptive_tol=True,
        verbose=True,
    )
    time_b = time.time() - t0_b

    # -------------------------------------------------------------------------
    # Comparison Summary
    # -------------------------------------------------------------------------
    speedup = time_a / time_b if time_b > 0 else 0.0
    fev_reduction = (res_a.n_fev - res_b.n_fev) / res_a.n_fev * 100.0 if res_a.n_fev > 0 else 0.0

    print("\n" + "=" * 72)
    print("  A/B BENCHMARK RESULTS SUMMARY (N=10 -> 20 -> 40, tol=2.0)")
    print("=" * 72)
    print(f"  {'Metric':<30} {'Case A (Fixed)':<18} {'Case B (Adaptive)':<18} {'Delta / Ratio'}")
    print("-" * 72)
    print(f"  {'Total Wall Time (s)':<30} {time_a:<18.2f} {time_b:<18.2f} {speedup:.2f}x speedup")
    print(f"  {'Final Stage Time (s)':<30} {res_a.wall_time_s:<18.2f} {res_b.wall_time_s:<18.2f} {res_a.wall_time_s / res_b.wall_time_s:.2f}x")
    print(f"  {'Final Stage ODE fev':<30} {res_a.n_fev:<18d} {res_b.n_fev:<18d} {fev_reduction:+.1f}%")
    print(f"  {'Final Stage Chunks':<30} {res_a.n_chunks:<18d} {res_b.n_chunks:<18d}")
    print(f"  {'Final Max Residual (J/kg-s)':<30} {res_a.max_residual:<18.3e} {res_b.max_residual:<18.3e}")
    print("-" * 72)
    print("  THERMODYNAMIC PARITY VERIFICATION:")
    print(f"  {'Thermal Efficiency (%)':<30} {res_a.eta * 100:<18.4f} {res_b.eta * 100:<18.4f} {abs(res_a.eta - res_b.eta) * 100:.2e} %")
    print(f"  {'Net Power (MW)':<30} {res_a.W_dot_net / 1e6:<18.5f} {res_b.W_dot_net / 1e6:<18.5f} {abs(res_a.W_dot_net - res_b.W_dot_net) / 1e3:.2f} kW")
    print(f"  {'Turbine Power (MW)':<30} {res_a.W_dot_turb / 1e6:<18.5f} {res_b.W_dot_turb / 1e6:<18.5f} {abs(res_a.W_dot_turb - res_b.W_dot_turb) / 1e3:.2f} kW")
    print(f"  {'Turbine Outlet T (K)':<30} {res_a.state.stations[6]['T']:<18.4f} {res_b.state.stations[6]['T']:<18.4f} {abs(res_a.state.stations[6]['T'] - res_b.state.stations[6]['T']):.4f} K")
    print(f"  {'Mixer Outlet T (K)':<30} {res_a.state.stations[12]['T']:<18.4f} {res_b.state.stations[12]['T']:<18.4f} {abs(res_a.state.stations[12]['T'] - res_b.state.stations[12]['T']):.4f} K")
    print(f"  {'Energy Balance Error (W)':<30} {res_a.state.energy_balance_err:<18.2f} {res_b.state.energy_balance_err:<18.2f}")
    print("=" * 72)


if __name__ == "__main__":
    run_ab_test()
