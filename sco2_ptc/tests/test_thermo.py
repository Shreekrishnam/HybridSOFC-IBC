"""
test_thermo.py
==============
Unit tests for thermodynamic property calculations via FluidProperties.
"""

from __future__ import annotations
import pytest
import numpy as np

from sco2_ptc.thermo import FluidProperties


@pytest.fixture
def co2_ethane_fluid():
    return FluidProperties(["CO2", "ETHANE"], [0.91, 0.09])


def test_critical_properties(co2_ethane_fluid):
    fluid = co2_ethane_fluid
    T_c, P_c, rho_c = fluid.get_critical_props()
    # Benchmark from EES and REFPROP: T_crit ~ 299.25 K, P_crit ~ 69.75 bar
    assert 295.0 < T_c < 305.0
    assert 65.0e5 < P_c < 75.0e5
    assert 400.0 < rho_c < 500.0


def test_reciprocity_pt_ph(co2_ethane_fluid):
    """Verifies that PT flash followed by PH flash returns identical state."""
    fluid = co2_ethane_fluid
    P_test = 100.0e5  # 100 bar
    T_test = 400.0    # 400 K

    pt_res = fluid.flash_pt(P_test, T_test)
    h_eval = pt_res["h"]

    ph_res = fluid.flash_ph(P_test, h_eval)
    assert abs(ph_res["T"] - T_test) < 0.01
    assert abs(ph_res["rho"] - pt_res["rho"]) / pt_res["rho"] < 1e-4


def test_cell_cache_consistency(co2_ethane_fluid):
    """Verifies that the per-cell cache returns identical values without drift."""
    fluid = co2_ethane_fluid
    cell_idx = 42
    P = 150.0e5
    H = 500000.0

    r1 = fluid.get_cell_props(cell_idx, P, H)
    r2 = fluid.get_cell_props(cell_idx, P, H)

    assert r1 is r2  # Same cached dictionary reference
    assert r1["T"] == r2["T"]

    # Changing H should trigger new calculation
    r3 = fluid.get_cell_props(cell_idx, P, H + 1000.0)
    assert r3["h"] != r1["h"]
