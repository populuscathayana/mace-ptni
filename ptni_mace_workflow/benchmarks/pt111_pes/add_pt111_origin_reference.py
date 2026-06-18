#!/usr/bin/env python
"""Add no-adatom origin-reference energies to an existing Pt(111) PES CSV.

The input CSV is expected to contain MACE and DFT energies for the 46-atom
adatom structures. This script reads the no-adatom 45-atom origin calculation,
computes one MACE single-point energy for that origin structure, and adds
origin-referenced adsorption-like energy columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def read_oszicar_e0(path: Path) -> float | None:
    if not path.exists():
        return None
    pattern = re.compile(rf"\bE0=\s*({FLOAT_RE})")
    value = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                value = float(match.group(1))
    return value


def read_outcar_energy(path: Path) -> float | None:
    if not path.exists():
        return None
    patterns = [
        re.compile(rf"energy\(sigma->0\)\s*=\s*({FLOAT_RE})"),
        re.compile(rf"free\s+energy\s+TOTEN\s*=\s*({FLOAT_RE})\s+eV"),
        re.compile(rf"energy\s+without\s+entropy\s*=\s*({FLOAT_RE})"),
    ]
    value = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    value = float(match.group(1))
                    break
    return value


def origin_dft_energy(origin_dir: Path, source: str) -> tuple[float, str]:
    if source in {"auto", "oszicar"}:
        value = read_oszicar_e0(origin_dir / "OSZICAR")
        if value is not None:
            return value, "OSZICAR:E0"
        if source == "oszicar":
            raise SystemExit(f"Could not read E0 from {origin_dir / 'OSZICAR'}")
    if source in {"auto", "outcar"}:
        value = read_outcar_energy(origin_dir / "OUTCAR")
        if value is not None:
            return value, "OUTCAR"
        if source == "outcar":
            raise SystemExit(f"Could not read energy from {origin_dir / 'OUTCAR'}")
    raise SystemExit(f"Could not determine DFT origin energy in {origin_dir}")


def compute_mace_energy(structure_path: Path, model: Path, device: str, default_dtype: str) -> tuple[float, int]:
    from ase.io import read
    from mace.calculators import MACECalculator

    atoms = read(str(structure_path), format="vasp")
    calc = MACECalculator(
        model_paths=str(model.resolve()),
        device=device,
        default_dtype=default_dtype,
    )
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    atoms.calc = None
    return energy, len(atoms)


def to_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def ensure_field(fieldnames: list[str], name: str) -> None:
    if name not in fieldnames:
        fieldnames.append(name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Add Pt(111) no-adatom origin-reference energy columns.")
    parser.add_argument("--csv", type=Path, required=True, help="Existing Pt(111) PES result CSV.")
    parser.add_argument("--origin-dir", type=Path, default=Path("mace_workspace/inputs/pt111/hex_point_origin"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--origin-structure",
        choices=["CONTCAR", "POSCAR"],
        default="CONTCAR",
        help="Origin structure used for the MACE single point. Default: CONTCAR.",
    )
    parser.add_argument("--dft-origin-source", choices=["auto", "oszicar", "outcar"], default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-dtype", default="float64")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    origin_dir = args.origin_dir.resolve()
    structure_path = origin_dir / args.origin_structure
    if not structure_path.exists():
        raise SystemExit(f"Origin structure does not exist: {structure_path}")

    dft_origin, dft_source = origin_dft_energy(origin_dir, args.dft_origin_source)
    mace_origin, origin_natoms = compute_mace_energy(structure_path, args.model, args.device, args.default_dtype)

    with args.csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise SystemExit(f"No rows in CSV: {args.csv}")

    new_fields = [
        "origin_label",
        "origin_structure",
        "origin_natoms",
        "dft_origin_energy_eV",
        "dft_origin_energy_source",
        "mace_origin_energy_eV",
        "mace_origin_model",
        "dft_origin_ref_eV",
        "dft_origin_ref_meV",
        "mace_origin_ref_eV",
        "mace_origin_ref_meV",
        "mace_minus_dft_origin_ref_eV",
        "mace_minus_dft_origin_ref_meV",
    ]
    for field in new_fields:
        ensure_field(fieldnames, field)

    for row in rows:
        dft_energy = to_float(row.get("dft_energy_eV"))
        mace_energy = to_float(row.get("mace_final_energy_eV"))
        if dft_energy is None:
            raise SystemExit(f"Missing dft_energy_eV for row {row.get('label')}")
        if mace_energy is None:
            raise SystemExit(f"Missing mace_final_energy_eV for row {row.get('label')}")
        dft_ref = dft_energy - dft_origin
        mace_ref = mace_energy - mace_origin
        diff = mace_ref - dft_ref
        row.update(
            {
                "origin_label": origin_dir.name,
                "origin_structure": str(structure_path),
                "origin_natoms": origin_natoms,
                "dft_origin_energy_eV": f"{dft_origin:.16g}",
                "dft_origin_energy_source": dft_source,
                "mace_origin_energy_eV": f"{mace_origin:.16g}",
                "mace_origin_model": str(args.model.resolve()),
                "dft_origin_ref_eV": f"{dft_ref:.16g}",
                "dft_origin_ref_meV": f"{dft_ref * 1000.0:.16g}",
                "mace_origin_ref_eV": f"{mace_ref:.16g}",
                "mace_origin_ref_meV": f"{mace_ref * 1000.0:.16g}",
                "mace_minus_dft_origin_ref_eV": f"{diff:.16g}",
                "mace_minus_dft_origin_ref_meV": f"{diff * 1000.0:.16g}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dft_values = [float(row["dft_origin_ref_meV"]) for row in rows]
    mace_values = [float(row["mace_origin_ref_meV"]) for row in rows]
    diff_values = [float(row["mace_minus_dft_origin_ref_meV"]) for row in rows]
    summary = {
        "input_csv": str(args.csv.resolve()),
        "output_csv": str(args.output.resolve()),
        "origin_dir": str(origin_dir),
        "origin_structure": str(structure_path),
        "origin_natoms": origin_natoms,
        "dft_origin_energy_eV": dft_origin,
        "dft_origin_energy_source": dft_source,
        "mace_origin_energy_eV": mace_origin,
        "mace_model": str(args.model.resolve()),
        "n_points": len(rows),
        "dft_origin_ref_min_meV": min(dft_values),
        "dft_origin_ref_max_meV": max(dft_values),
        "mace_origin_ref_min_meV": min(mace_values),
        "mace_origin_ref_max_meV": max(mace_values),
        "diff_min_meV": min(diff_values),
        "diff_max_meV": max(diff_values),
    }

    summary_path = args.summary or args.output.with_suffix(".origin_reference.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"DFT origin ({dft_source}): {dft_origin:.10f} eV")
    print(f"MACE origin ({args.origin_structure}, {origin_natoms} atoms): {mace_origin:.10f} eV")
    print(f"Rows: {len(rows)}")
    print(f"Output CSV: {args.output.resolve()}")
    print(f"Summary JSON: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
