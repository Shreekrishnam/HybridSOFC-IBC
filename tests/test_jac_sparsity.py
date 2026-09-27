"""Guards the declared Jacobian sparsity pattern against a dense finite difference.

An entry that is nonzero in reality but absent from the declared pattern is worse
than a missing optimisation: SciPy's num_jac groups columns by graph colour and
attributes each group's difference to the declared columns only, so an undeclared
coupling is silently folded into an unrelated entry. The MUSCL stencil widened the
inlet coupling of every block and this is what checks that the pattern kept up.

The finite difference is taken with freeze_U=True because that is the condition
holding during every Jacobian column: RecuperatorBlock freezes the cell pressure
profile there, which is what removes the dense upstream pressure chain from the
pattern in the first place.
"""

import numpy as np
import pytest

from sco2_ptc.config import CycleConfig, CycleDOF
from sco2_ptc.flowsheet import Flowsheet
from sco2_ptc.thermo import FluidProperties


N_CELLS = 8
REL_TOL = 1e-6  # column entries below this fraction of the column max are noise


def _dense_fd_jacobian(fs: Flowsheet, y: np.ndarray) -> np.ndarray:
    """Dense forward-difference Jacobian evaluated the way num_jac would see it."""
    # Prime the frozen caches with an unfrozen pass, then hold them fixed.
    fs.fluid.clear_cell_cache()
    fs.rhs(0.0, y, freeze_U=False)
    f0 = fs.rhs(0.0, y, freeze_U=True)

    n = y.size
    J = np.empty((n, n))
    for j in range(n):
        dh = max(abs(y[j]), 1.0) * 1e-6
        y_p = y.copy()
        y_p[j] += dh
        J[:, j] = (fs.rhs(0.0, y_p, freeze_U=True) - f0) / dh
    return J


@pytest.mark.parametrize("use_muscl", [False, True])
def test_declared_sparsity_covers_every_real_coupling(use_muscl):
    cfg = CycleConfig(use_legacy_p_in=True, use_muscl=use_muscl)
    dof = CycleDOF(A_tube=4.20, r=0.50, frac_recomp=0.40, N_cells=N_CELLS)
    fluid = FluidProperties(cfg.fluid_components, [cfg.x_CO2, 1.0 - cfg.x_CO2])

    fs = Flowsheet(cfg, dof, fluid)
    y = fs.build_initial_guess()

    J = _dense_fd_jacobian(fs, y)
    declared = fs.jac_sparsity.toarray() != 0.0

    # Scale each row by its own largest entry so that a genuinely coupled but
    # weak row is not compared against the largest coupling in the whole matrix.
    row_scale = np.maximum(np.abs(J).max(axis=1, keepdims=True), 1e-30)
    significant = (np.abs(J) / row_scale) > REL_TOL

    missing = significant & ~declared
    if missing.any():
        rows, cols = np.nonzero(missing)
        detail = ", ".join(
            f"({r},{c}) rel={abs(J[r, c]) / row_scale[r, 0]:.2e}"
            for r, c in list(zip(rows, cols))[:20]
        )
        pytest.fail(
            f"{missing.sum()} real coupling(s) absent from the declared sparsity "
            f"(use_muscl={use_muscl}): {detail}"
        )


@pytest.mark.parametrize("use_muscl", [False, True])
def test_pattern_is_not_trivially_dense(use_muscl):
    """A pattern that declares everything would pass the test above vacuously."""
    cfg = CycleConfig(use_legacy_p_in=True, use_muscl=use_muscl)
    dof = CycleDOF(A_tube=4.20, r=0.50, frac_recomp=0.40, N_cells=N_CELLS)
    fluid = FluidProperties(cfg.fluid_components, [cfg.x_CO2, 1.0 - cfg.x_CO2])

    fs = Flowsheet(cfg, dof, fluid)
    n = fs.n_states
    density = fs.jac_sparsity.nnz / float(n * n)
    assert density < 0.25, f"sparsity pattern is {density:.1%} dense; coloring will not help"
