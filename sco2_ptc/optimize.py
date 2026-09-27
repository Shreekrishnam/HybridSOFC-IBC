"""
optimize.py
===========
Optimization and multi-parameter sweep for the sCO2 recompression cycle:
- Optimizes recuperator area fraction `r` and recompression fraction `frac_recomp`
- Multi-dimensional continuation sweeps across (r, frac_recomp, A_tube)
- Warm-starting cache to accelerate successive evaluations
"""

from __future__ import annotations
import sys
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize

# Ensure parent python directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sco2_ptc.config import CycleConfig, CycleDOF
from sco2_ptc.thermo import FluidProperties
from sco2_ptc.solver import solve_cycle_ptc, CycleResult


class CycleOptimizer:
    """Manages cycle optimization and continuation sweeps with state caching."""

    def __init__(
        self,
        cfg: Optional[CycleConfig] = None,
        fluid: Optional[FluidProperties] = None,
        N_cells: int = 40,
        conv_tol: float = 5.0
    ):
        self.cfg = cfg or CycleConfig()
        self.fluid = fluid or FluidProperties(self.cfg.fluid_components, self.cfg.mole_fractions)
        self.N_cells = N_cells
        self.conv_tol = conv_tol

        # Warm-start cache: keyed by tuple of rounded decision variables
        self.cache: Dict[Tuple[float, float, float], np.ndarray] = {}
        self.last_successful_y: Optional[np.ndarray] = None

    def evaluate(self, r: float, frac_recomp: float, A_tube: float = 4.2) -> Tuple[float, CycleResult]:
        """
        Evaluates the cycle at (r, frac_recomp, A_tube).
        Returns (eta, CycleResult).
        """
        dof = CycleDOF(A_tube=A_tube, r=r, frac_recomp=frac_recomp, N_cells=self.N_cells)

        # Look for closest warm-start point in cache
        warm_y = None
        key = (round(r, 2), round(frac_recomp, 2), round(A_tube, 1))
        if key in self.cache:
            warm_y = self.cache[key]
        elif self.last_successful_y is not None:
            warm_y = self.last_successful_y

        res = solve_cycle_ptc(
            dof=dof,
            cfg=self.cfg,
            fluid=self.fluid,
            y0=warm_y,
            tau_max=300.0,
            chunk_dtau=25.0,
            conv_tol=self.conv_tol,
            verbose=False,
        )

        if res.converged:
            self.last_successful_y = res.y_final
            self.cache[key] = res.y_final

        return res.eta, res

    def optimize_2d(
        self,
        r_init: float = 0.5,
        frac_init: float = 0.4,
        A_tube: float = 4.2,
        r_bounds: Tuple[float, float] = (0.2, 0.8),
        frac_bounds: Tuple[float, float] = (0.1, 0.6),
        method: str = "Nelder-Mead",
        max_iter: int = 30
    ) -> Dict[str, Any]:
        """
        Optimizes (r, frac_recomp) to maximize cycle thermal efficiency.
        """
        print("=" * 68)
        print("  Starting 2D Cycle Optimization (Maximize Thermal Efficiency)")
        print(f"  Initial guess: r={r_init:.3f}, frac_recomp={frac_init:.3f}")
        print("=" * 68)

        eval_count = [0]
        best_eta = [-1.0]
        best_point = [None]

        def objective(x: np.ndarray) -> float:
            r_val, f_val = x[0], x[1]
            eval_count[0] += 1

            # Penalty for out of bounds
            if not (r_bounds[0] <= r_val <= r_bounds[1] and frac_bounds[0] <= f_val <= frac_bounds[1]):
                return 1.0  # Large penalty

            t0 = time.time()
            eta, res = self.evaluate(r=r_val, frac_recomp=f_val, A_tube=A_tube)
            dt = time.time() - t0

            # Pinch penalty if pinch temperature is violated
            pinch_penalty = 0.0
            min_pinch = min(res.state.htr_state.dT_min, res.state.ltr_state.dT_min)
            if min_pinch < 2.0:
                pinch_penalty = 0.05 * (2.0 - min_pinch)

            obj_val = -eta + pinch_penalty

            if eta > best_eta[0]:
                best_eta[0] = eta
                best_point[0] = (r_val, f_val, res)
                marker = " ** Best **"
            else:
                marker = ""

            print(f"  Iter {eval_count[0]:>2d}: r={r_val:.4f}, frac={f_val:.4f} -> eta={eta*100:.3f}% (pinch={min_pinch:.1f} K, {dt:.1f}s){marker}")
            return obj_val

        x0 = np.array([r_init, frac_init])
        bounds = [r_bounds, frac_bounds]

        opt_res = minimize(
            objective,
            x0=x0,
            bounds=bounds if method in ("L-BFGS-B", "SLSQP") else None,
            method=method,
            options={"maxiter": max_iter, "disp": True}
        )

        r_opt, frac_opt = opt_res.x[0], opt_res.x[1]
        print("=" * 68)
        print(f"  Optimization Finished: r* = {r_opt:.4f}, frac_recomp* = {frac_opt:.4f}")
        print(f"  Max Efficiency: {best_eta[0]*100:.3f} %")
        print("=" * 68)

        return {
            "r_opt": r_opt,
            "frac_recomp_opt": frac_opt,
            "eta_opt": best_eta[0],
            "opt_result": opt_res,
            "best_cycle_result": best_point[0][2] if best_point[0] else None,
        }

    def parametric_sweep(
        self,
        r_list: List[float],
        frac_list: List[float],
        A_tube: float = 4.2
    ) -> List[Dict[str, Any]]:
        """
        Executes a 2D continuation parametric sweep across (r, frac_recomp).
        """
        print("=" * 68)
        print(f"  Parametric Sweep: {len(r_list)} x {len(frac_list)} = {len(r_list)*len(frac_list)} points")
        print("=" * 68)

        results = []
        for r_val in r_list:
            for f_val in frac_list:
                t0 = time.time()
                eta, res = self.evaluate(r=r_val, frac_recomp=f_val, A_tube=A_tube)
                dt = time.time() - t0
                print(f"  r={r_val:.2f}, frac={f_val:.2f} -> eta={eta*100:.3f}%, W_net={res.W_dot_net/1e6:.3f} MW ({dt:.1f}s)")
                results.append({
                    "r": r_val,
                    "frac_recomp": f_val,
                    "eta": eta,
                    "W_dot_net": res.W_dot_net,
                    "W_dot_turb": res.W_dot_turb,
                    "Q_dot_in": res.Q_dot_in,
                    "converged": res.converged,
                    "dT_min_htr": res.state.htr_state.dT_min,
                    "dT_min_ltr": res.state.ltr_state.dT_min,
                })
        return results


if __name__ == "__main__":
    opt = CycleOptimizer(N_cells=30, conv_tol=5.0)
    opt.optimize_2d(r_init=0.5, frac_init=0.4, max_iter=15)
