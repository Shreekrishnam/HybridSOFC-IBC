"""
io.py
=====
Serialization and deserialization for converged PTC cycle solutions and exergy analyses.
Saves complete state vectors, HX spatial profiles, station properties, and metadata
to compact, cross-platform compressed NumPy archives (.npz).
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .config import CycleConfig, CycleDOF
from .solver import CycleResult
from .flowsheet import FlowsheetCycleState
from .components.recuperator import RecuperatorState
from .exergy import CycleExergyAnalysis, ExergyComponentResult


def save_cycle_result(
    result: CycleResult,
    filepath: str | Path,
    exergy: Optional[CycleExergyAnalysis] = None,
) -> Path:
    """
    Saves a converged CycleResult and optional CycleExergyAnalysis to a compressed .npz archive.

    Parameters:
    -----------
    result : CycleResult
    filepath : str or Path (will append .npz if missing)
    exergy : CycleExergyAnalysis, optional
    """
    filepath = Path(filepath)
    if not filepath.name.endswith(".npz"):
        filepath = filepath.with_suffix(".npz")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    s = result.state
    htr = s.htr_state
    ltr = s.ltr_state

    # Construct JSON-serializable metadata dictionary
    meta: Dict[str, Any] = {
        "converged": bool(result.converged),
        "tau_final": float(result.tau_final),
        "max_residual": float(result.max_residual),
        "wall_time_s": float(result.wall_time_s),
        "n_chunks": int(result.n_chunks),
        "n_fev": int(result.n_fev),
        "dof": {
            "A_tube": float(result.dof.A_tube),
            "r": float(result.dof.r),
            "frac_recomp": float(result.dof.frac_recomp),
            "N_cells": int(result.dof.N_cells),
        },
        "performance": {
            "eta": float(result.eta),
            "W_dot_net": float(result.W_dot_net),
            "W_dot_turb": float(result.W_dot_turb),
            "W_dot_comp_1": float(s.W_dot_comp_1),
            "W_dot_comp_2": float(s.W_dot_comp_2),
            "Q_dot_in": float(result.Q_dot_in),
            "Q_dot_out": float(s.Q_dot_out),
            "energy_balance_err": float(s.energy_balance_err),
        },
        "recuperators": {
            "HTR": {
                "Q_total": float(htr.Q_total),
                "dT_min": float(htr.dT_min),
                "dT_avg": float(htr.dT_avg),
                "dP_cold": float(htr.dP_cold),
                "dP_hot": float(htr.dP_hot),
            },
            "LTR": {
                "Q_total": float(ltr.Q_total),
                "dT_min": float(ltr.dT_min),
                "dT_avg": float(ltr.dT_avg),
                "dP_cold": float(ltr.dP_cold),
                "dP_hot": float(ltr.dP_hot),
            },
        },
        "stations": {str(k): v for k, v in s.stations.items()},
    }

    if exergy is not None:
        meta["exergy"] = {
            "T0": float(exergy.T0),
            "P0": float(exergy.P0),
            "S_dot_gen_total": float(exergy.S_dot_gen_total),
            "E_dot_dest_total": float(exergy.E_dot_dest_total),
            "eta_exergetic": float(exergy.eta_exergetic),
            "components": {
                name: {
                    "S_dot_gen": float(c.S_dot_gen),
                    "E_dot_dest": float(c.E_dot_dest),
                    "dest_fraction": float(c.dest_fraction),
                }
                for name, c in exergy.components.items()
            },
        }

    # Save arrays + JSON string into .npz
    np.savez_compressed(
        filepath,
        metadata_json=json.dumps(meta, indent=2),
        y_final=result.y_final,
        res_history=np.array(result.res_history),
        tau_history=np.array(result.tau_history),
        # HTR spatial profiles
        htr_H_cold=htr.H_cold,
        htr_H_hot=htr.H_hot,
        htr_T_cold=htr.T_cold,
        htr_T_hot=htr.T_hot,
        htr_P_cold=htr.P_cold,
        htr_P_hot=htr.P_hot,
        htr_rho_cold=htr.rho_cold,
        htr_rho_hot=htr.rho_hot,
        htr_U_cell=htr.U_cell,
        htr_Q_cell=htr.Q_cell,
        htr_dT_cell=htr.dT_cell,
        # LTR spatial profiles
        ltr_H_cold=ltr.H_cold,
        ltr_H_hot=ltr.H_hot,
        ltr_T_cold=ltr.T_cold,
        ltr_T_hot=ltr.T_hot,
        ltr_P_cold=ltr.P_cold,
        ltr_P_hot=ltr.P_hot,
        ltr_rho_cold=ltr.rho_cold,
        ltr_rho_hot=ltr.rho_hot,
        ltr_U_cell=ltr.U_cell,
        ltr_Q_cell=ltr.Q_cell,
        ltr_dT_cell=ltr.dT_cell,
    )
    return filepath


def load_cycle_result(
    filepath: str | Path,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Loads a saved cycle simulation from a .npz archive.

    Returns:
    --------
    metadata : Dict[str, Any] (all cycle performance, station, and exergy metrics)
    arrays : Dict[str, np.ndarray] (state vectors, history, and HX spatial profiles)
    """
    filepath = Path(filepath)
    if not filepath.exists() and not filepath.name.endswith(".npz"):
        filepath = filepath.with_suffix(".npz")

    data = np.load(filepath)
    meta = json.loads(str(data["metadata_json"]))
    arrays = {key: data[key] for key in data.files if key != "metadata_json"}

    return meta, arrays
