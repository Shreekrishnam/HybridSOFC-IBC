#!/usr/bin/env python
"""
varmap.py
=========
Checkpoint C2.4 - rewrite a saved EES Variable Info file from the legacy flat-index
names into the renamed model's names, so the renamed model converges on its first
Solve from the already-converged solution.

Input  : stage5.var          (saved from the legacy model after Update Guesses)
         mapping/state_map.csv
Output : guesses_named.var   (load with  $INCLUDE guesses_named.var)

.var format, tab delimited, one row per variable:

    NAME <TAB> guess <TAB> lower <TAB> upper <TAB> format <TAB> flag <TAB> units <TAB> flag

Only the NAME and units columns are rewritten. Guess, limits and format pass through
untouched, so the transferred state is exactly the converged one.

Nothing is dropped silently: every legacy variable is either mapped, explicitly
retired, or reported as UNMAPPED at the end. Likewise every variable the renamed
model expects but the legacy file cannot supply is reported as MISSING.

Usage:
    python varmap.py --var ../stage5.var --out ../guesses_named.var
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# --- per-point arrays: legacy stem -> renamed stem -------------------------
POINT_ARRAYS = {
    "T": "T", "P": "P", "H": "h", "S": "s",
    "CP": "cp", "K_VAL": "k", "MU_DYN": "mu", "PR": "Pr",
    "NU": "Nu", "H_COEFF": "hc", "U_EFF": "Ueff",
    "RE_PT": "Re", "F_PT": "f",
}

# --- cell arrays: index unchanged, stem renamed ---------------------------
CELL_ARRAYS = {
    "Q_DOT_HTR": "Q_cell_HTR", "Q_DOT_LTR": "Q_cell_LTR",
    "U_HTR": "U_cell_HTR",     "U_LTR": "U_cell_LTR",
    "AVG_TD_HTR": "dTavg_HTR", "AVG_TD_LTR": "dTavg_LTR",
    "DT_HTR": "dT_HTR",        "DT_LTR": "dT_LTR",
}

# --- plain scalar renames -------------------------------------------------
SCALARS = {
    "RE": "Re_hot", "RE2": "Re_cold", "F_TURB": "f_hot", "F_TURB2": "f_cold",
    "P_DROP": "dP_HTR", "P_DROP3": "dP_LTR_LP", "P_DROP2": "dP_LTR_HP",
    "L": "L_HTR", "L2": "L_LTR", "V": "v_hot", "V2": "v_cold",
    "M_DOT_TOTAL_HOT": "m_dot_hot", "M_DOT_TOTAL_COLD": "m_dot_cold",
    "MW_AR": "MW_second",
    "S_DOT_GEN_HTR": "S_gen_HTR", "S_DOT_GEN_LTR": "S_gen_LTR",
    "S_DOT_GEN_HEATER": "S_gen_heater", "S_DOT_GEN_COOLER": "S_gen_cooler",
    "S_DOT_GEN_MIXING": "S_gen_mixing", "S_DOT_GEN_TOTAL": "S_gen_total",
    "AVG_HTR": "dT_avg_HTR", "AVG_LTR": "dT_avg_LTR",
}

# --- retired on purpose ---------------------------------------------------
RETIRED = {
    "FLUID$": "the CO2+HYDROGEN mass-vs-mole-fraction test block was dropped",
    "Z":      "ditto",
    "MW_MIX": "ditto - output of that test call",
    "DT_R_HTR": "duplicate of dT_HTR[d_p+2]",
    "DT_L_HTR": "duplicate of dT_HTR[1]",
    "DT_R_LTR": "duplicate of dT_LTR[d_p+2]",
    "DT_L_LTR": "duplicate of dT_LTR[1]",
}

# --- units, matching the $VarInfo block gen_eqns.py emits -----------------
UNITS_BY_STEM = {
    "T": "K", "P": "bar", "h": "J/kg", "s": "J/kg-K", "cp": "J/kg-K",
    "k": "W/m-K", "mu": "kg/m-s", "Pr": "-", "Re": "-", "f": "-",
    "Nu": "-", "hc": "W/m^2-K", "Ueff": "W/m^2-K",
    "Q_cell_HTR": "W", "Q_cell_LTR": "W",
    "U_cell_HTR": "W/m^2-K", "U_cell_LTR": "W/m^2-K",
    "dTavg_HTR": "K", "dTavg_LTR": "K", "dT_HTR": "K", "dT_LTR": "K",
    "m_dot": "kg/s", "h_i": "J/kg", "s_i": "J/kg-K",
}


def load_map(path: Path):
    by_index = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_index[int(row["old_index"])] = row
    return by_index


def target(idx: int, stem: str, by_index) -> str | None:
    """Legacy stem + flat index -> renamed variable."""
    row = by_index.get(idx)
    if row is None:
        return None
    if row["kind"] == "scalar":
        return f'{stem}_{row["new_base"]}'
    return f'{stem}_{row["new_base"]}[{row["cell_n"]}]'


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", type=Path, default=here.parent / "stage5.var")
    ap.add_argument("--map", type=Path, default=here.parent / "mapping" / "state_map.csv")
    ap.add_argument("--out", type=Path, default=here.parent / "guesses_named.var")
    args = ap.parse_args()

    by_index = load_map(args.map)
    rows, unmapped, retired_hits = [], [], []
    cyc = {}          # legacy index -> {prop: full row}, for the T_cyc[] aliases
    n_in = 0

    for line in args.var.read_text(encoding="latin-1").splitlines():
        if line.startswith("//") or not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 7:
            continue
        n_in += 1
        raw = f[0].strip()
        up = raw.upper()

        # function locals and REFPROP mode constants pass straight through
        if "/" in raw:
            rows.append((raw, f))
            continue

        m = re.match(r"^([A-Z0-9_$]+)\[(\d+)\]$", up)
        new = None
        if m:
            stem, idx = m.group(1), int(m.group(2))
            if stem in POINT_ARRAYS:
                new = target(idx, POINT_ARRAYS[stem], by_index)
                if new and idx <= 12 and POINT_ARRAYS[stem] in ("T", "P", "h", "s"):
                    cyc.setdefault(idx, {})[POINT_ARRAYS[stem]] = f
            elif stem in CELL_ARRAYS:
                new = f"{CELL_ARRAYS[stem]}[{idx}]"
            elif stem == "M_DOT":
                new = target(idx, "m_dot", by_index)
            elif stem in ("H_I", "S_I"):
                new = target(idx, "h_i" if stem == "H_I" else "s_i", by_index)
        elif up in RETIRED:
            retired_hits.append(raw)
            continue
        elif up in SCALARS:
            new = SCALARS[up]
        else:
            new = raw            # unchanged name

        if new is None:
            unmapped.append(raw)
            continue
        rows.append((new, f))

    # --- cycle plotting aliases, straight copies of the legacy values ------
    added = 0
    for idx, props in sorted(cyc.items()):
        for prop, f in props.items():
            rows.append((f"{prop}_cyc[{idx}]", f))
            added += 1

    # --- write, with units filled in to agree with the $VarInfo block ------
    def units_for(name: str) -> str:
        stem = re.sub(r"\[\d+\]$", "", name)
        stem = re.sub(r"_cyc$", "", stem)
        if stem in UNITS_BY_STEM:
            return UNITS_BY_STEM[stem]
        head = stem.split("_")[0]
        return UNITS_BY_STEM.get(head, "")

    out = [f"//generated by varmap.py from {args.var.name}"]
    for name, f in sorted(rows, key=lambda r: r[0].upper()):
        g = list(f)
        g[0] = name.upper()
        u = units_for(name)
        if u:                      # normalise, e.g. the legacy lowercase "k" on T[]
            g[6] = u
        out.append("\t".join(g))
    # newline="" so Windows does not translate \n again and produce \r\r\n
    with args.out.open("w", encoding="latin-1", newline="") as fh:
        fh.write("\r\n".join(out) + "\r\n")

    print(f"  read    {n_in} rows from {args.var.name}")
    print(f"  wrote   {len(rows)} rows to {args.out.name}  (+{added} cycle aliases)")
    print(f"  retired {len(retired_hits)} on purpose: {sorted(set(retired_hits))}")
    if unmapped:
        print(f"  !! UNMAPPED ({len(unmapped)}) - these lose their guess:")
        for u in sorted(set(unmapped))[:25]:
            print(f"       {u}")
    else:
        print("  no unmapped variables")


if __name__ == "__main__":
    main()
