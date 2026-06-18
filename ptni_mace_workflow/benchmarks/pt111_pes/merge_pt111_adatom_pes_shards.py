#!/usr/bin/env python
"""Merge sharded Pt(111) adatom PES CSV files."""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path


def to_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def point_sort_key(row: dict) -> tuple[int, str]:
    try:
        return int(float(row.get("point_index", "999999999"))), row.get("label", "")
    except ValueError:
        return 999999999, row.get("label", "")


def ensure_field(fieldnames: list[str], name: str) -> None:
    if name not in fieldnames:
        fieldnames.append(name)


def recompute_global_relative_energies(rows: list[dict], fieldnames: list[str]) -> None:
    mace_energies = [to_float(row.get("mace_final_energy_eV")) for row in rows]
    dft_energies = [to_float(row.get("dft_energy_eV")) for row in rows]
    mace_valid = [value for value in mace_energies if value is not None]
    dft_valid = [value for value in dft_energies if value is not None]
    if not mace_valid:
        return

    for name in [
        "mace_rel_eV",
        "mace_rel_meV",
        "dft_rel_eV",
        "dft_rel_meV",
        "mace_minus_dft_rel_eV",
        "mace_minus_dft_rel_meV",
    ]:
        ensure_field(fieldnames, name)

    mace_min = min(mace_valid)
    dft_min = min(dft_valid) if dft_valid else None
    for row in rows:
        mace_energy = to_float(row.get("mace_final_energy_eV"))
        dft_energy = to_float(row.get("dft_energy_eV"))
        if mace_energy is not None:
            mace_rel = mace_energy - mace_min
            row["mace_rel_eV"] = f"{mace_rel:.16g}"
            row["mace_rel_meV"] = f"{mace_rel * 1000.0:.16g}"
        if dft_energy is not None and dft_min is not None:
            dft_rel = dft_energy - dft_min
            row["dft_rel_eV"] = f"{dft_rel:.16g}"
            row["dft_rel_meV"] = f"{dft_rel * 1000.0:.16g}"
        mace_rel_value = to_float(row.get("mace_rel_eV"))
        dft_rel_value = to_float(row.get("dft_rel_eV"))
        if mace_rel_value is not None and dft_rel_value is not None:
            diff = mace_rel_value - dft_rel_value
            row["mace_minus_dft_rel_eV"] = f"{diff:.16g}"
            row["mace_minus_dft_rel_meV"] = f"{diff * 1000.0:.16g}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge sharded Pt(111) PES result CSVs.")
    parser.add_argument("--input-glob", required=True, help="Glob for shard CSV files.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument(
        "--no-recompute-rel",
        action="store_true",
        help="Keep shard-local relative energy columns instead of recomputing global relative energies.",
    )
    args = parser.parse_args()

    paths = sorted(Path(path) for path in glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No shard CSV files matched: {args.input_glob}")

    rows: list[dict] = []
    fieldnames: list[str] = []
    seen: dict[str, Path] = {}

    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            for row in reader:
                label = row.get("label", "")
                if label and label in seen and not args.allow_duplicates:
                    raise SystemExit(f"Duplicate label {label!r} in {path} and {seen[label]}")
                if label:
                    seen[label] = path
                rows.append(row)

    rows.sort(key=point_sort_key)
    if not args.no_recompute_rel:
        recompute_global_relative_energies(rows, fieldnames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Shard files: {len(paths)}")
    print(f"Rows merged: {len(rows)}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
