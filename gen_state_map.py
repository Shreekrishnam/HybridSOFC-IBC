#!/usr/bin/env python
"""
gen_state_map.py
================
Generate the canonical index -> name map for the sCO2 recompression cycle model.

The legacy model indexes every state point in one flat array: cycle states 1..12,
then four recuperator sides at 12 + k*d_p + n (k = 0..3, n = 1..d_p), numbered
left-to-right in the diagram rather than in flow direction. See EES model.docx and
the State indexing section of CLAUDE.md.

This script is the single source of truth for the rename. It emits
mapping/state_map.csv, which C2.4 uses to rewrite guess values from the old .var
into the renamed model.

Usage:
    python gen_state_map.py --d-p 80 --out ../mapping/state_map.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# --- cycle states 1..12 ---------------------------------------------------
# name, component, stream, description
CYCLE_STATES = {
    1:  ("comp1_in",    "cooler/compressor-1", "cold", "Cooler outlet = main compressor inlet (T = T_cold)"),
    2:  ("comp1_out",   "compressor-1/LTR",    "cold", "Main compressor outlet = LTR high-pressure inlet (P = P_H)"),
    3:  ("comp2_out",   "compressor-2/mix",    "cold", "Recompressor outlet, into the mixing junction"),
    4:  ("HTR_HP_out",  "HTR/heater",          "hot",  "HTR high-pressure outlet = heater inlet"),
    5:  ("turb_in",     "heater/turbine",      "hot",  "Turbine inlet (T = T_hot)"),
    6:  ("turb_out",    "turbine/HTR",         "hot",  "Turbine outlet = HTR low-pressure inlet (P = P_L)"),
    7:  ("HTR_LP_out",  "HTR/LTR",             "hot",  "HTR low-pressure outlet = LTR low-pressure inlet"),
    8:  ("LTR_LP_out",  "LTR/split",           "hot",  "LTR low-pressure outlet = flow split point"),
    9:  ("comp2_in",    "split/compressor-2",  "cold", "Recompressor inlet branch (frac_recomp of the flow)"),
    10: ("cooler_in",   "split/cooler",        "cold", "Cooler inlet branch"),
    11: ("LTR_HP_out",  "LTR/mix",             "cold", "LTR high-pressure outlet, into the mixing junction"),
    12: ("mix_out",     "mix/HTR",             "hot",  "Mixing junction outlet = HTR high-pressure inlet"),
}

# --- the four recuperator sides, in 12 + k*d_p + n order -------------------
# k, array base, stream, flow-direction note
SIDES = [
    (0, "HTR_HP", "cold", "high-pressure side of the high-temperature recuperator; n increases along the flow"),
    (1, "HTR_LP", "hot",  "low-pressure side of the high-temperature recuperator; n increases against the flow"),
    (2, "LTR_HP", "cold", "high-pressure side of the low-temperature recuperator; n increases along the flow"),
    (3, "LTR_LP", "hot",  "low-pressure side of the low-temperature recuperator; n increases against the flow"),
]


def build(d_p: int) -> list[dict]:
    rows = []
    for idx in sorted(CYCLE_STATES):
        name, comp, stream, desc = CYCLE_STATES[idx]
        rows.append({
            "old_index": idx,
            "kind": "scalar",
            "new_base": name,
            "cell_n": "",
            "component": comp,
            "stream": stream,
            "description": desc,
        })
    for k, base, stream, desc in SIDES:
        for n in range(1, d_p + 1):
            rows.append({
                "old_index": 12 + k * d_p + n,
                "kind": "array",
                "new_base": base,
                "cell_n": n,
                "component": base.split("_")[0],
                "stream": stream,
                "description": f"{desc} (point {n} of {d_p})",
            })
    return rows


def verify(rows: list[dict], d_p: int) -> None:
    n_expected = 12 + 4 * d_p
    idx = [r["old_index"] for r in rows]
    assert len(idx) == n_expected, f"row count {len(idx)} != {n_expected}"
    assert sorted(idx) == list(range(1, n_expected + 1)), "indices are not a clean 1..N partition"
    # round-trip: name -> index must be unique and invertible
    seen = {}
    for r in rows:
        key = (r["new_base"], r["cell_n"])
        assert key not in seen, f"duplicate name {key}"
        seen[key] = r["old_index"]
    print(f"  verified: {n_expected} indices, no gaps, no duplicates, round-trip OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-p", type=int, default=80)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "mapping" / "state_map.csv")
    args = ap.parse_args()

    rows = build(args.d_p)
    verify(rows, args.d_p)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {args.out}  (d_p = {args.d_p})")

    print("\n  sample:")
    for r in rows[:3] + rows[11:15] + rows[-2:]:
        tgt = r["new_base"] if r["kind"] == "scalar" else f'{r["new_base"]}[{r["cell_n"]}]'
        print(f'    T[{r["old_index"]:>3}]  ->  T_{tgt}')


if __name__ == "__main__":
    main()
