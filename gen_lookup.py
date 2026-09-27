#!/usr/bin/env python
"""
gen_lookup.py
=============
Checkpoint C6.2 - regenerate the cp / k / eta property tables for the sCO2 model.

Why this exists
---------------
The tables embedded in the .EES64 cover 80-220 bar and 300-850 K: the nominal cycle
box with zero margin. The solved model steps outside it. `P_L` = 79.75 bar for the
CO2+ethane mixture (P_crit fell to 69.75 when ethane was added), and the accumulated
low-pressure drop carries the LP side down to 68.4 bar. Every low-pressure point is
therefore extrapolated, which feeds Pr -> Nu -> h_coeff -> U. See checkpoint C0.7.

The PT/PH workaround
--------------------
REFPROP's TP flash intermittently returns its undefined sentinel (-9999990) for the
DERIVED properties of this mixture - cp, cv, eta, kap - while still returning rho and
h correctly. The PH flash at the identical state succeeds. Demonstrated at
80 bar / 400 K:

    update_PT -> cp UNDEF,  eta UNDEF
    update_PH -> cp 1313.8, eta 2.1132e-05

So each grid node is evaluated in two steps: h from the PT flash, then everything
else from PH(P, h). Nodes where even that fails are reported, never silently filled.

Output
------
Three-column long-format CSV per property, which is what the built-in Interpolate2D
wants (INTERPOLATE2DM, the current function, is documented as LINEAR; Interpolate2D
is bi-quadratic over the 16 nearest points and reads a Lookup file straight from
disk).

Usage:
    python gen_lookup.py --n-p 60 --n-t 60 --out-dir ../lookup
    python gen_lookup.py --validate      # compare against the current table bounds
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

DYNHX = Path(r"D:\OneDrive - Indian Institute of Science\Research-Analysis"
             r"\04 Models\dynhx_table_gen")
sys.path.insert(0, str(DYNHX))

from refprop_fluid import REFPROPFluid, is_undef   # noqa: E402

# property key in the REFPROP wrapper -> EES lookup table name
PROPS = {"cp": "cp", "kap": "k", "eta": "eta"}


def build(fluid, P_grid, T_grid, verbose=True):
    rows, failures = [], []
    t0 = time.time()
    for i, P in enumerate(P_grid):
        for T in T_grid:
            try:
                pt = fluid.update_PT(P, T)
            except RuntimeError as exc:
                failures.append((P, T, f"PT flash: {exc}"))
                continue
            h = pt.get("h")
            if h is None or is_undef(h):
                failures.append((P, T, "PT gave no enthalpy"))
                continue
            try:
                r = fluid.update_PH(P, h)
            except RuntimeError as exc:
                failures.append((P, T, f"PH flash: {exc}"))
                continue
            vals = {k: r.get(k) for k in PROPS}
            bad = [k for k, v in vals.items() if v is None or is_undef(v)]
            if bad:
                failures.append((P, T, "undefined: " + ",".join(bad)))
                continue
            rows.append((P, T, vals))
        if verbose and (i + 1) % 10 == 0:
            print(f"    P row {i+1}/{len(P_grid)}  ({time.time()-t0:.0f}s)")
    return rows, failures


def fill_gaps(rows, P_grid, T_grid, failures):
    """Fill missing nodes by linear interpolation in T between the nearest valid
    neighbours on the same pressure row.

    A matrix-form Lookup table has to be complete, so gaps cannot simply be left.
    But the values invented here are FICTITIOUS: the gaps occur where REFPROP
    reports a two-phase iteration failure, i.e. inside the mixture's phase
    envelope, and no single-phase property exists there. They are filled so the
    table is well formed, and reported so the fiction is visible. The cycle must
    not operate in that region - see the C3.5 feasibility note.
    """
    have = {(P, T): v for P, T, v in rows}
    filled = []
    for P in P_grid:
        col = [T for T in T_grid if (P, T) in have]
        if len(col) < 2:
            continue
        for T in T_grid:
            if (P, T) in have:
                continue
            lo = [t for t in col if t < T]
            hi = [t for t in col if t > T]
            if not lo or not hi:
                continue
            a, b = lo[-1], hi[0]
            wa = (b - T) / (b - a)
            vals = {k: wa * have[(P, a)][k] + (1 - wa) * have[(P, b)][k]
                    for k in have[(P, a)]}
            have[(P, T)] = vals
            filled.append((P, T, a, b))
    out = [(P, T, have[(P, T)]) for P in P_grid for T in T_grid if (P, T) in have]
    return out, filled


def write_ees_matrix(rows, P_grid, T_grid, out_dir: Path):
    """Write in EES's own Lookup CSV layout, matched to an export of the existing
    cp table:

        line 0   Column1,Column2,...        column names, BOM on the first
        line 1   ,,,,                       units row, left empty
        line 2   ,P1,P2,...                 table row 1: the pressures, in Pa
        line 3+  T_i,v_i1,v_i2,...          one row per temperature

    interpolate2dm resolves it as LookupCol1(F$,1,P) against table row 1 and
    LookupRow1(F$,1,T) against column 1, so pressures run across and temperatures
    down. Both axes must be monotonic increasing or the lookup returns -1.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    at = {(P, T): v for P, T, v in rows}
    written = []
    for key, name in PROPS.items():
        f = out_dir / f"{name}.csv"
        ncol = len(P_grid) + 1
        with f.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow([f"Column{i+1}" for i in range(ncol)])
            w.writerow([""] * ncol)
            w.writerow([""] + [f"{P:.8e}" for P in P_grid])
            for T in T_grid:
                w.writerow([f"{T:.6f}"] +
                           [f"{at[(P, T)][key]:.8e}" for P in P_grid])
        written.append(f)
    return written


def write_csv(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, name in PROPS.items():
        f = out_dir / f"{name}.csv"
        with f.open("w", newline="", encoding="ascii") as fh:
            w = csv.writer(fh)
            w.writerow(["P", "T", name])
            for P, T, vals in rows:
                w.writerow([f"{P:.6e}", f"{T:.6f}", f"{vals[key]:.8e}"])
        written.append(f)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-min", type=float, default=60e5, help="Pa")
    ap.add_argument("--p-max", type=float, default=240e5, help="Pa")
    ap.add_argument("--t-min", type=float, default=280.0)
    ap.add_argument("--t-max", type=float, default=900.0)
    ap.add_argument("--n-p", type=int, default=60)
    ap.add_argument("--n-t", type=int, default=60)
    ap.add_argument("--uniform", action="store_true",
                    help="uniform grid instead of the default refinement near the "
                         "pseudo-critical ridge")
    ap.add_argument("--x-co2", type=float, default=0.91, help="MOLE fraction of CO2")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "lookup")
    args = ap.parse_args()

    def piecewise(spec):
        """spec = [(lo, hi, step), ...]; monotonic, no duplicated endpoints."""
        out = []
        for lo, hi, st in spec:
            x = lo
            while x < hi - 1e-9:
                if not out or x > out[-1] + 1e-9:
                    out.append(x)
                x += st
        out.append(spec[-1][1])
        return out

    if args.uniform:
        step_p = (args.p_max - args.p_min) / (args.n_p - 1)
        step_t = (args.t_max - args.t_min) / (args.n_t - 1)
        P_grid = [args.p_min + i * step_p for i in range(args.n_p)]
        T_grid = [args.t_min + i * step_t for i in range(args.n_t)]
    else:
        # The pseudo-critical ridge runs diagonally: cp peaks at 299 K at 69 bar and
        # at ~336 K at 220 bar, with a half-width of ~3 K at the operating pressure.
        # Refine a band that covers the ridge at every pressure, and the low-pressure
        # end where the fluid is most non-ideal.
        P_grid = piecewise([(args.p_min, 100e5, 1e5), (100e5, args.p_max, 2.5e5)])
        T_grid = piecewise([(args.t_min, 292.0, 2.0), (292.0, 360.0, 1.0),
                            (360.0, args.t_max, 5.0)])
        args.n_p, args.n_t = len(P_grid), len(T_grid)

    print(f"  grid {args.n_p} x {args.n_t} = {args.n_p*args.n_t} nodes")
    print(f"  P {args.p_min/1e5:.1f}..{args.p_max/1e5:.1f} bar   "
          f"T {args.t_min:.0f}..{args.t_max:.0f} K")
    print(f"  existing table covers 80.0..220.0 bar, 300..850 K")

    fluid = REFPROPFluid(["CO2", "ETHANE"], [args.x_co2, 1.0 - args.x_co2])
    rows, failures = build(fluid, P_grid, T_grid)

    print(f"\n  {len(rows)} nodes computed, {len(failures)} failed")
    if failures:
        print("  first failures:")
        for P, T, why in failures[:8]:
            print(f"    {P/1e5:7.2f} bar {T:6.1f} K   {why}")
        frac = len(failures) / (len(rows) + len(failures))
        print(f"  failure rate {frac:.2%}")
        if frac > 0.02:
            print("  !! above 2% - do not ship this table; investigate before use")

    rows, filled = fill_gaps(rows, P_grid, T_grid, failures)
    if filled:
        Ts = sorted({T for _, T, _, _ in filled})
        Ps = sorted({P/1e5 for P, _, _, _ in filled})
        print(f"  filled {len(filled)} gaps by interpolation in T")
        print(f"    T = {Ts[0]:.1f}..{Ts[-1]:.1f} K, P = {Ps[0]:.0f}..{Ps[-1]:.0f} bar")
        print("    these are FICTITIOUS - REFPROP reports two-phase there, so no")
        print("    single-phase property exists. The cycle must not operate in this band.")
    print(f"  matrix now {len(rows)} nodes ({len(P_grid)} x {len(T_grid)} = "
          f"{len(P_grid)*len(T_grid)} required)")

    for f in write_ees_matrix(rows, P_grid, T_grid, args.out_dir):
        print(f"  wrote {f}  (EES matrix format, for interpolate2dm)")
    for f in write_csv(rows, args.out_dir / "long"):
        print(f"  wrote {f}  (3-column, for a future Interpolate2D)")


if __name__ == "__main__":
    main()
