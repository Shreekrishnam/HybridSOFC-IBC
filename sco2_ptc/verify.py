"""
verify.py
=========
Verification and parity benchmarking script:
Compares the Python PTC solver solution against the EES reference solution
in refactor_corr_rho_st2.var.
"""

from __future__ import annotations
import sys
from pathlib import Path
import time
import numpy as np

# Ensure parent python directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sco2_ptc.config import CycleConfig, CycleDOF
from sco2_ptc.thermo import FluidProperties
from sco2_ptc.solver import solve_cycle_ptc, solve_cycle_ladder


def load_ees_reference_var(var_file_path: Path) -> dict[str, float]:
    """Reads key scalar values from an EES .var file."""
    ref = {}
    with open(var_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                name = parts[0].strip().upper()
                try:
                    val = float(parts[1].strip())
                    ref[name] = val
                except ValueError:
                    pass
    return ref


def verify_against_ees(var_path: Path | None = None, N_cells: int = 80, adaptive_tol: bool = False) -> bool:
    if var_path is None:
        var_path = Path(__file__).resolve().parent.parent.parent / "refactor_corr_rho_st2.var"

    print("=" * 72)
    print("  SUPERCRITICAL CO2 RECOMPRESSION BRAYTON CYCLE")
    print(f"  Verification: Python PTC vs EES (refactor_corr_rho_st2.var) [adaptive_tol={adaptive_tol}]")
    print("=" * 72)

    if not var_path.exists():
        print(f"  Warning: {var_path} not found.")
        ref = {
            "ETA": 0.4631063,
            "W_DOT_NET": 1.487237e6,
            "W_DOT_TURB": 2.055287e6,
            "W_DOT_COMP_1": 1.935361e5,
            "W_DOT_COMP_2": 3.745130e5,
            "Q_DOT_IN": 3.211438e6,
            "T_COMP1_OUT": 325.0478,
            "T_TURB_OUT": 733.3982,
            "T_MIX_OUT": 457.8288,
        }
    else:
        ref = load_ees_reference_var(var_path)

    # Use legacy P_in for bit-exact comparison against C5.1
    cfg = CycleConfig(use_legacy_p_in=True)
    dof = CycleDOF(A_tube=4.2, r=0.5, frac_recomp=0.4, N_cells=N_cells)
    fluid = FluidProperties(cfg.fluid_components, cfg.mole_fractions)

    t0 = time.time()
    if N_cells > 10:
        res = solve_cycle_ladder(
            target_dof=dof,
            cfg=cfg,
            fluid=fluid,
            conv_tol=2.0,
            adaptive_tol=adaptive_tol,
            verbose=True,
        )
    else:
        res = solve_cycle_ptc(
            dof=dof,
            cfg=cfg,
            fluid=fluid,
            tau_max=600.0,
            chunk_dtau=25.0,
            conv_tol=2.0,
            adaptive_tol=adaptive_tol,
            verbose=True,
        )
    wall_time = time.time() - t0

    print(res.summary())

    st = res.state.stations
    comparisons = [
        ("Thermal Efficiency (eta)", ref.get("ETA", 0.4631063), res.eta, 5e-4, "abs"),
        ("Net Power (W_dot_net, MW)", ref.get("W_DOT_NET", 1.487237e6) / 1e6, res.W_dot_net / 1e6, 1e-3, "rel"),
        ("Turbine Power (W_turb, MW)", ref.get("W_DOT_TURB", 2.055287e6) / 1e6, res.W_dot_turb / 1e6, 1e-3, "rel"),
        ("Heat In (Q_dot_in, MW)", ref.get("Q_DOT_IN", 3.211438e6) / 1e6, res.Q_dot_in / 1e6, 1e-3, "rel"),
        ("Comp 1 Outlet T (T2, K)", ref.get("T_COMP1_OUT", 325.0478), st[2]["T"], 0.2, "abs"),
        ("Turbine Outlet T (T6, K)", ref.get("T_TURB_OUT", 733.3982), st[6]["T"], 0.2, "abs"),
        ("Mixer Outlet T (T12, K)", ref.get("T_MIX_OUT", 457.8288), st[12]["T"], 0.5, "abs"),
    ]

    print("\n" + "=" * 72)
    print(f"  {'Metric':<28} {'EES Target':>12} {'Python PTC':>12} {'Delta':>10} {'Status':>8}")
    print("-" * 72)

    all_pass = True
    for name, ees_val, py_val, tol, mode in comparisons:
        diff = abs(py_val - ees_val)
        err = diff if mode == "abs" else (diff / abs(ees_val))
        passed = err <= tol
        if not passed:
            all_pass = False
        status_str = "PASS" if passed else "FAIL"
        print(f"  {name:<28} {ees_val:>12.4f} {py_val:>12.4f} {diff:>10.4e} {status_str:>8}")

    print("-" * 72)
    min_pinch_htr = res.state.htr_state.dT_min
    min_pinch_ltr = res.state.ltr_state.dT_min
    pinch_ok = min_pinch_htr > 0.0 and min_pinch_ltr > 0.0
    print(f"  Pinch HTR: {min_pinch_htr:.2f} K (> 0), LTR: {min_pinch_ltr:.2f} K (> 0) -> {'PASS' if pinch_ok else 'FAIL'}")
    print(f"  Overall Verification Result: {'PASSED' if (all_pass and pinch_ok) else 'FAILED'}")
    print("=" * 72)

    return all_pass and pinch_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify Python PTC against EES model")
    parser.add_argument("--N", type=int, default=80, help="Discretization resolution (cells per side)")
    parser.add_argument("--adaptive-tol", action="store_true", help="Enable adaptive Newton/BDF error tolerance")
    args = parser.parse_args()
    verify_against_ees(N_cells=args.N, adaptive_tol=args.adaptive_tol)
