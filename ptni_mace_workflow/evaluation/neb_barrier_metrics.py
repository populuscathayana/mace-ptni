#!/usr/bin/env python
"""Summarize NEB barrier errors from an extxyz prediction file.

Expected input: one extxyz file containing reference energies under REF_energy
and predicted energies under a key such as MACE_energy. For each NEB group, the
script keeps the final frame of each image and compares max(E) - E(initial).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def get_info_float(atoms, key: str) -> float | None:
    value = atoms.info.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute NEB barrier MAE from predicted extxyz.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--group-key", default="source_group")
    parser.add_argument("--image-key", default="neb_image")
    parser.add_argument("--frame-key", default="frame_index")
    parser.add_argument("--ref-energy-key", default="REF_energy")
    parser.add_argument("--pred-energy-key", default="MACE_energy")
    args = parser.parse_args()

    from ase.io import iread

    groups = defaultdict(list)
    for atoms in iread(args.input.as_posix(), index=":"):
        if args.image_key not in atoms.info:
            continue
        group = str(atoms.info.get(args.group_key, atoms.info.get("source_path", "ungrouped")))
        groups[group].append(atoms)

    rows = []
    for group, atoms_list in groups.items():
        by_image = {}
        for atoms in atoms_list:
            image = int(atoms.info[args.image_key])
            frame = int(atoms.info.get(args.frame_key, 0))
            if image not in by_image or frame > int(by_image[image].info.get(args.frame_key, 0)):
                by_image[image] = atoms

        if len(by_image) < 3:
            continue

        image_ids = sorted(by_image)
        ref = [get_info_float(by_image[i], args.ref_energy_key) for i in image_ids]
        pred = [get_info_float(by_image[i], args.pred_energy_key) for i in image_ids]
        if any(x is None for x in ref) or any(x is None for x in pred):
            continue

        ref0 = ref[0]
        pred0 = pred[0]
        ref_rel = [e - ref0 for e in ref]
        pred_rel = [e - pred0 for e in pred]
        ref_barrier = max(ref_rel)
        pred_barrier = max(pred_rel)
        ref_ts_image = image_ids[ref_rel.index(ref_barrier)]
        pred_ts_image = image_ids[pred_rel.index(pred_barrier)]
        rows.append(
            {
                "source_group": group,
                "n_images": len(image_ids),
                "ref_barrier_eV": ref_barrier,
                "pred_barrier_eV": pred_barrier,
                "error_eV": pred_barrier - ref_barrier,
                "abs_error_eV": abs(pred_barrier - ref_barrier),
                "ref_ts_image": ref_ts_image,
                "pred_ts_image": pred_ts_image,
            }
        )

    if not rows:
        raise SystemExit("No complete NEB groups with both reference and predicted energies were found.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mae = sum(row["abs_error_eV"] for row in rows) / len(rows)
    print(f"Wrote {len(rows)} NEB barrier rows to {args.out_csv}")
    print(f"Barrier MAE: {mae:.6f} eV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
