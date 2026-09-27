"""Run the pure sCO2 optimization as a standalone campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sco2_ptc.fluids import CAMPAIGN_FLUIDS
from sco2_ptc.mixture_optimization import MixtureOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure sCO2 optimization for the supercritical Brayton cycle"
    )
    parser.add_argument("--n-cells", type=int, default=40)
    parser.add_argument("--no-muscl", action="store_true")
    parser.add_argument(
        "--limiter",
        choices=["van_leer", "minmod", "van_albada"],
        default="van_leer",
    )
    parser.add_argument("--tol", type=float, default=2.0)
    parser.add_argument("--xatol", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=25)
    args = parser.parse_args()

    optimizer = MixtureOptimizer(
        N_cells=args.n_cells,
        conv_tol=args.tol,
        use_muscl=not args.no_muscl,
        muscl_limiter=args.limiter,
        n_workers=1,
        save_dir=Path(__file__).resolve().parent.parent.parent
        / "results"
        / "pure_co2_optimization",
        xatol=args.xatol,
        max_iter=args.max_iter,
    )
    optimizer.run_all_mixtures([CAMPAIGN_FLUIDS[0]])


if __name__ == "__main__":
    main()
