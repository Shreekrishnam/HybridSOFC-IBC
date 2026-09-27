"""
test_continuation.py
====================
Unit tests for continuation stepping and warm-start robustness.
"""

from __future__ import annotations
import pytest
import numpy as np

from sco2_ptc.config import CycleConfig, CycleDOF
from sco2_ptc.thermo import FluidProperties
from sco2_ptc.solver import solve_cycle_ptc


def test_warm_start_acceleration():
    cfg = CycleConfig(use_legacy_p_in=True)
    fluid = FluidProperties(cfg.fluid_components, cfg.mole_fractions)

    # First point: N=10 cold solve
    dof1 = CycleDOF(r=0.5, frac_recomp=0.4, N_cells=10)
    res1 = solve_cycle_ptc(
        dof=dof1, cfg=cfg, fluid=fluid, tau_max=300.0, chunk_dtau=20.0, conv_tol=20.0, verbose=False
    )
    assert res1.max_residual < 30.0

    # Second point: step r from 0.5 -> 0.55 using res1 as warm start
    dof2 = CycleDOF(r=0.55, frac_recomp=0.38, N_cells=10)
    res2 = solve_cycle_ptc(
        dof=dof2,
        cfg=cfg,
        fluid=fluid,
        y0=res1.y_final,
        tau_max=200.0,
        chunk_dtau=20.0,
        conv_tol=20.0,
        verbose=False,
    )

    assert res2.converged or res2.max_residual < 25.0
    # Warm start should settle rapidly
    assert res2.wall_time_s < res1.wall_time_s or res2.n_chunks <= res1.n_chunks
