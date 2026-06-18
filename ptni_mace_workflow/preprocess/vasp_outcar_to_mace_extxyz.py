#!/usr/bin/env python
"""Convert VASP OUTCAR trajectories to MACE-ready extended XYZ.

The script reads every ionic step from each OUTCAR with ASE, copies the DFT
energy and forces into explicit REF_* fields, and adds grouping metadata for
group-aware train/validation/test splits.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path


def relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def discover_outcars(root: Path) -> list[Path]:
    candidates = []
    for path in root.rglob("*"):
        if path.is_file() and path.name.upper() == "OUTCAR":
            candidates.append(path)
    return sorted(candidates)


def infer_config_type(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    joined = "/".join(parts)
    if "cineb" in joined or "ci-neb" in joined or "neb" in joined:
        return "neb"
    if re.fullmatch(r"\d{2,3}", path.parent.name):
        return "neb"
    if "bulk" in joined or "crystal" in joined:
        return "bulk"
    if "slab" in joined or "surface" in joined:
        return "slab"
    return "Default"


def infer_source_group(path: Path, root: Path) -> str:
    if re.fullmatch(r"\d{2,3}", path.parent.name):
        return relpath(path.parent.parent, root)
    return relpath(path.parent, root)


def infer_neb_image(path: Path) -> int | None:
    if re.fullmatch(r"\d{2,3}", path.parent.name):
        return int(path.parent.name)
    return None


def read_outcar(path: Path):
    from ase.io import read

    # ASE's VASP OUTCAR reader accepts ":" for all ionic steps.
    configs = read(path.as_posix(), index=":", format="vasp-out")
    if not isinstance(configs, list):
        configs = [configs]
    return configs


def get_energy_and_forces(atoms):
    import numpy as np

    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if forces.shape != (len(atoms), 3):
        raise ValueError(f"unexpected forces shape {forces.shape}")
    if not math.isfinite(energy) or not np.isfinite(forces).all():
        raise ValueError("non-finite energy or forces")
    return energy, forces


def maybe_copy_stress(atoms):
    import numpy as np

    try:
        stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception:
        return None
    if stress.shape == (6,) and np.isfinite(stress).all():
        return stress
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert VASP OUTCAR files into MACE-ready extxyz."
    )
    parser.add_argument("--root", type=Path, required=True, help="Root containing OUTCAR files.")
    parser.add_argument("--out", type=Path, required=True, help="Output extxyz path.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV with one row per retained configuration.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth ionic step from each OUTCAR. Default: 1.",
    )
    parser.add_argument(
        "--last-only",
        action="store_true",
        help="Keep only the final ionic step from each OUTCAR.",
    )
    parser.add_argument(
        "--max-force",
        type=float,
        default=None,
        help="Drop frames whose maximum atomic force norm exceeds this value in eV/Angstrom.",
    )
    parser.add_argument(
        "--include-stress",
        action="store_true",
        help="Also write REF_stress when ASE can read it. Usually leave off for slab/vacuum data.",
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    root = args.root.resolve()
    outcars = discover_outcars(root)
    if not outcars:
        raise SystemExit(f"No OUTCAR files found under {root}")

    retained = []
    rows = []
    rejected = Counter()
    type_counts = Counter()

    for outcar in outcars:
        try:
            configs = read_outcar(outcar)
        except Exception as exc:
            print(f"[skip] failed to read {outcar}: {exc}", file=sys.stderr)
            rejected["read_error"] += 1
            continue

        if args.last_only and configs:
            selected = [(len(configs) - 1, configs[-1])]
        else:
            selected = [(i, atoms) for i, atoms in enumerate(configs) if i % args.stride == 0]

        config_type = infer_config_type(outcar)
        source_path = relpath(outcar, root)
        source_group = infer_source_group(outcar, root)
        neb_image = infer_neb_image(outcar)

        for frame_index, atoms in selected:
            try:
                energy, forces = get_energy_and_forces(atoms)
            except Exception as exc:
                print(f"[skip] {outcar} frame {frame_index}: {exc}", file=sys.stderr)
                rejected["missing_energy_or_forces"] += 1
                continue

            force_norms = np.linalg.norm(forces, axis=1)
            max_force = float(force_norms.max()) if len(force_norms) else 0.0
            if args.max_force is not None and max_force > args.max_force:
                rejected["max_force_filter"] += 1
                continue

            atoms = atoms.copy()
            atoms.info["REF_energy"] = energy
            atoms.arrays["REF_forces"] = forces
            atoms.info["config_type"] = config_type
            atoms.info["source_path"] = source_path
            atoms.info["source_group"] = source_group
            atoms.info["frame_index"] = int(frame_index)
            atoms.info["max_force_eVA"] = max_force
            atoms.info["natoms"] = int(len(atoms))
            if neb_image is not None:
                atoms.info["neb_image"] = int(neb_image)
            if args.include_stress:
                stress = maybe_copy_stress(atoms)
                if stress is not None:
                    atoms.info["REF_stress"] = stress
            atoms.calc = None

            retained.append(atoms)
            type_counts[config_type] += 1
            rows.append(
                {
                    "source_path": source_path,
                    "source_group": source_group,
                    "config_type": config_type,
                    "neb_image": "" if neb_image is None else neb_image,
                    "frame_index": frame_index,
                    "natoms": len(atoms),
                    "REF_energy_eV": energy,
                    "max_force_eVA": max_force,
                }
            )

    if not retained:
        raise SystemExit(f"No configurations retained. Rejections: {dict(rejected)}")

    from ase.io import write

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out.as_posix(), retained, format="extxyz")

    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(retained)} configurations to {args.out}")
    print(f"Config types: {dict(type_counts)}")
    print(f"Rejected: {dict(rejected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
