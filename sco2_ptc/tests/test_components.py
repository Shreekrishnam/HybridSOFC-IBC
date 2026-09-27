"""
test_components.py
==================
Unit tests for cycle components: turbomachinery, mixer, heater, and recuperator.
"""

from __future__ import annotations
import pytest
import numpy as np

from sco2_ptc.thermo import FluidProperties
from sco2_ptc.components.turbomachinery import solve_compressor, solve_turbine
from sco2_ptc.components.junctions import solve_mixer, solve_heater, solve_cooler
from sco2_ptc.components.recuperator import RecuperatorBlock


@pytest.fixture
def fluid():
    return FluidProperties(["CO2", "ETHANE"], [0.91, 0.09])


def test_compressor_thermodynamics(fluid):
    P_in = 80.0e5
    H_in = 300000.0
    P_out = 220.0e5
    eta = 0.88

    H_out, T_out, w_act, out_props = solve_compressor(P_in, H_in, P_out, eta, fluid)

    in_props = fluid.flash_ph(P_in, H_in)
    # Compression raises temperature and pressure
    assert T_out > in_props["T"]
    assert H_out > H_in
    assert w_act > 0.0
    # Second law: actual outlet entropy must exceed inlet entropy
    assert out_props["s"] > in_props["s"]


def test_turbine_thermodynamics(fluid):
    P_in = 220.0e5
    H_in = 1100000.0
    P_out = 80.0e5
    eta = 0.93

    H_out, T_out, w_act, out_props = solve_turbine(P_in, H_in, P_out, eta, fluid)

    in_props = fluid.flash_ph(P_in, H_in)
    # Expansion drops temperature and enthalpy
    assert T_out < in_props["T"]
    assert H_out < H_in
    assert w_act > 0.0
    # Second law: actual outlet entropy must exceed inlet entropy
    assert out_props["s"] > in_props["s"]


def test_mixer_energy_balance():
    H_cold = 400000.0
    m_cold = 8.37
    H_recomp = 500000.0
    m_recomp = 5.58

    H_mix = solve_mixer(H_cold, m_cold, H_recomp, m_recomp)

    # Check conservation: m_cold * H_cold + m_recomp * H_recomp == m_tot * H_mix
    m_tot = m_cold + m_recomp
    energy_in = m_cold * H_cold + m_recomp * H_recomp
    energy_out = m_tot * H_mix
    assert abs(energy_in - energy_out) < 1e-8


def test_recuperator_block(fluid):
    """Verifies that the recuperator residual computation produces consistent duties."""
    N = 10
    block = RecuperatorBlock(
        name="TestRecup",
        N=N,
        L=30.0,
        D=0.02,
        A_cell=1.0,
        A_flow_tube=0.04,
        V_cell=0.1,
        R_cond=0.008 / 22.0,
        m_dot_tube_cold=0.05,
        m_dot_tube_hot=0.09,
        m_dot_cold=8.0,
        m_dot_hot=14.0,
    )

    H_c = np.linspace(350000.0, 500000.0, N)
    H_h = np.linspace(450000.0, 650000.0, N)

    dH_c, dH_h, state = block.compute_residuals(
        H_c_cells=H_c,
        H_h_cells=H_h,
        H_c_in=320000.0,
        P_c_in=220.0e5,
        H_h_in=680000.0,
        P_h_in=80.0e5,
        fluid=fluid,
    )

    assert len(dH_c) == N
    assert len(dH_h) == N
    assert state.Q_total > 0.0
    assert state.dP_cold > 0.0
    assert state.dP_hot > 0.0
