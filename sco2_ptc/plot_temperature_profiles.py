"""
plot_temperature_profiles.py
============================
Solves the supercritical cycle to steady-state convergence, extracts local
temperature and pressure distributions along the HTR and LTR recuperators,
computes post-convergence component entropy generation, and saves publication-quality plots.
"""

import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure parent python directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sco2_ptc.config import CycleConfig, CycleDOF
from sco2_ptc.thermo import FluidProperties
from sco2_ptc.solver import solve_cycle_ladder
from sco2_ptc.exergy import compute_cycle_exergy


def generate_profiles_and_exergy(N_cells: int = 40, output_path: Path | None = None):
    print("=" * 76)
    print(f"  SOLVING CYCLE FOR TEMPERATURE PROFILES & EXERGY ANALYSIS (N={N_cells})")
    print("=" * 76)

    cfg = CycleConfig(use_legacy_p_in=True)
    dof = CycleDOF(A_tube=4.20, r=0.50, frac_recomp=0.40, N_cells=N_cells)
    fluid = FluidProperties(cfg.fluid_components, cfg.mole_fractions)

    ladder = [10, 20, N_cells] if N_cells > 20 else [10, N_cells]
    t0 = time.time()
    res = solve_cycle_ladder(
        target_dof=dof,
        cfg=cfg,
        fluid=fluid,
        ladder=ladder,
        conv_tol=2.0,
        adaptive_tol=False,
        verbose=True,
    )
    wall_time = time.time() - t0
    print(f"\nSolve completed in {wall_time:.2f} s. Max residual: {res.max_residual:.3e} J/(kg s)")

    # -------------------------------------------------------------------------
    # Post-Convergence Second-Law Exergy & Entropy Generation Analysis
    # -------------------------------------------------------------------------
    exergy_res = compute_cycle_exergy(res.state, fluid, T0=298.15)
    print("\n" + exergy_res.summary())

    # -------------------------------------------------------------------------
    # Extract Recuperator Spatial Profiles
    # -------------------------------------------------------------------------
    htr = res.state.htr_state
    ltr = res.state.ltr_state

    # Normalized axial coordinate z/L from 0 (cold fluid inlet) to 1 (cold fluid outlet)
    # Cold stream enters at z=0 and exits at z=1.
    # Hot stream enters at z=1 and exits at z=0 (counterflow).
    z_norm = np.linspace(0.0, 1.0, N_cells)

    # Note on orientation:
    # Cell i=0 is cold inlet; Cell i=N-1 is cold outlet.
    # Counterflow hot stream flows from cell N-1 to 0.
    T_htr_c = htr.T_cold
    T_htr_h = htr.T_hot
    dT_htr = htr.dT_cell

    T_ltr_c = ltr.T_cold
    T_ltr_h = ltr.T_hot
    dT_ltr = ltr.dT_cell

    # -------------------------------------------------------------------------
    # Publication-Grade 4-Panel Plot
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # 1. HTR Temperature Profile
    ax1 = axes[0, 0]
    ax1.plot(z_norm, T_htr_h, color="#d95f02", lw=2.5, label="Hot Stream ($T_{hot}$)")
    ax1.plot(z_norm, T_htr_c, color="#1f78b4", lw=2.5, label="Cold Stream ($T_{cold}$)")
    min_idx_h = np.argmin(dT_htr)
    ax1.scatter([z_norm[min_idx_h]], [T_htr_c[min_idx_h]], color="red", zorder=5, s=60)
    ax1.annotate(
        f"Pinch: $\\Delta T_{{min}} = {htr.dT_min:.2f}$ K",
        xy=(z_norm[min_idx_h], T_htr_c[min_idx_h]),
        xytext=(z_norm[min_idx_h] - 0.25, T_htr_c[min_idx_h] + 35),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=6),
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fffae6", ec="#cc9900")
    )
    ax1.set_title("High-Temperature Recuperator (HTR) Temperature Profile", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Normalized Length ($z/L$)", fontsize=11)
    ax1.set_ylabel("Temperature [K]", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. LTR Temperature Profile
    ax2 = axes[0, 1]
    ax2.plot(z_norm, T_ltr_h, color="#e7298a", lw=2.5, label="Hot Stream ($T_{hot}$)")
    ax2.plot(z_norm, T_ltr_c, color="#33a02c", lw=2.5, label="Cold Stream ($T_{cold}$)")
    min_idx_l = np.argmin(dT_ltr)
    ax2.scatter([z_norm[min_idx_l]], [T_ltr_c[min_idx_l]], color="red", zorder=5, s=60)
    ax2.annotate(
        f"Pinch: $\\Delta T_{{min}} = {ltr.dT_min:.2f}$ K",
        xy=(z_norm[min_idx_l], T_ltr_c[min_idx_l]),
        xytext=(z_norm[min_idx_l] - 0.25, T_ltr_c[min_idx_l] + 25),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=6),
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fffae6", ec="#cc9900")
    )
    ax2.set_title("Low-Temperature Recuperator (LTR) Temperature Profile", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Normalized Length ($z/L$)", fontsize=11)
    ax2.set_ylabel("Temperature [K]", fontsize=11)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Local Temperature Difference (Pinch Evolution)
    ax3 = axes[1, 0]
    ax3.plot(z_norm, dT_htr, color="#7570b3", lw=2.2, label=f"HTR $\\Delta T$ (min={htr.dT_min:.2f} K)")
    ax3.plot(z_norm, dT_ltr, color="#1b9e77", lw=2.2, label=f"LTR $\\Delta T$ (min={ltr.dT_min:.2f} K)")
    ax3.axhline(0, color="gray", linestyle=":", lw=1.2)
    ax3.set_title("Local Stream Temperature Difference $\\Delta T(z) = T_{hot} - T_{cold}$", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Normalized Length ($z/L$)", fontsize=11)
    ax3.set_ylabel("$\\Delta T$ [K]", fontsize=11)
    ax3.legend(loc="best", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # 4. Component Exergy Destruction Breakdown
    ax4 = axes[1, 1]
    comp_names = list(exergy_res.components.keys())
    e_dests_kw = [c.E_dot_dest / 1e3 for c in exergy_res.components.values()]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(comp_names)))
    bars = ax4.barh(comp_names, e_dests_kw, color=colors, edgecolor="black", alpha=0.85)
    ax4.set_xlabel("Exergy Destruction Rate $\\dot{E}_{dest}$ [kW]", fontsize=11)
    ax4.set_title(f"Cycle Exergy Destruction Breakdown (Total: {exergy_res.E_dot_dest_total/1e3:.1f} kW)", fontsize=12, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.6, axis="x")
    for bar in bars:
        w = bar.get_width()
        pct = (w / (exergy_res.E_dot_dest_total / 1e3)) * 100.0
        ax4.text(w + max(e_dests_kw) * 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.1f} kW ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold")
    ax4.set_xlim(0, max(e_dests_kw) * 1.3)

    plt.tight_layout()

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "hx_temperature_distribution.png"

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSuccessfully generated and saved plot to: {output_path}")
    plt.close(fig)

    # Also save copy to brain artifact directory if specified
    brain_dir = Path(r"C:\Users\Jaichander S\.gemini\antigravity-cli\brain\971e5329-d20b-49da-9774-2ac7ce357799")
    if brain_dir.exists():
        brain_fig_path = brain_dir / "hx_temperature_distribution.png"
        fig.savefig(brain_fig_path, dpi=200, bbox_inches="tight")
        print(f"Artifact copy saved to: {brain_fig_path}")

    return res, exergy_res


if __name__ == "__main__":
    generate_profiles_and_exergy(N_cells=40)
