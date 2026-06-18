#!/usr/bin/env python
"""Convert collected OUTCAR files to MACE-ready extxyz files.

Each input file keeps its naming style: for example
OUTCAR_public_home_calc_001_OUTCAR.OUTCAR becomes
OUTCAR_public_home_calc_001_OUTCAR.extxyz.

The script writes DFT labels as REF_energy and REF_forces, which can be used by
MACE with:
  --energy_key REF_energy --forces_key REF_forces
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath, PureWindowsPath


def load_manifest(path: Path | None) -> dict[str, str]:
    """Return target filename -> original absolute path from collect_outcars.py."""
    if path is None:
        return {}
    mapping = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target_name = row.get("target_name")
            source_abs_path = row.get("source_abs_path")
            target_abs_path = row.get("target_abs_path")
            if target_name and source_abs_path:
                mapping[target_name] = source_abs_path
            if target_abs_path and source_abs_path:
                mapping[Path(target_abs_path).name] = source_abs_path
    return mapping


def split_path_parts(path_text: str) -> list[str]:
    return [part for part in re.split(r"[\\/]+", path_text) if part]


def infer_config_type(path_text: str) -> str:
    lower = path_text.lower()
    if "cineb" in lower or "ci-neb" in lower or "neb" in lower:
        return "neb"
    if re.search(r"(^|[_/\\])0?\d{1,2}([_/\\]|$)", lower) and "outcar" in lower:
        if "neb" in lower or "ts" in lower:
            return "neb"
    if "bulk" in lower or "crystal" in lower:
        return "bulk"
    if "slab" in lower or "slabd" in lower or "surface" in lower:
        return "slab"
    return "Default"


def infer_neb_image(path_text: str) -> int | None:
    parts = split_path_parts(path_text)
    if len(parts) >= 2 and parts[-1].upper() == "OUTCAR" and re.fullmatch(r"\d{1,3}", parts[-2]):
        return int(parts[-2])
    return None


def infer_source_group(path_text: str, fallback_name: str) -> str:
    parts = split_path_parts(path_text)
    if len(parts) >= 3 and parts[-1].upper() == "OUTCAR" and re.fullmatch(r"\d{1,3}", parts[-2]):
        return "/".join(parts[:-2])
    if len(parts) >= 2 and parts[-1].upper() == "OUTCAR":
        return "/".join(parts[:-1])
    return Path(fallback_name).stem


def output_name_for(source: Path) -> str:
    if source.suffix:
        return f"{source.stem}.extxyz"
    return f"{source.name}.extxyz"


def finite_energy_forces(atoms):
    import numpy as np

    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if forces.shape != (len(atoms), 3):
        raise ValueError(f"unexpected forces shape {forces.shape}")
    if not math.isfinite(energy) or not np.isfinite(forces).all():
        raise ValueError("non-finite energy or forces")
    return energy, forces


def maybe_stress(atoms):
    import numpy as np

    try:
        stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception:
        return None
    if stress.shape == (6,) and np.isfinite(stress).all():
        return stress
    return None


def read_selected_frames(path: Path, stride: int, last_only: bool):
    from ase.io import iread, read

    if last_only:
        atoms = read(path.as_posix(), index=-1, format="vasp-out")
        return [(-1, atoms)]

    selected = []
    for index, atoms in enumerate(iread(path.as_posix(), index=":", format="vasp-out")):
        if index % stride == 0:
            selected.append((index, atoms))
    return selected


def convert_one(task: dict) -> dict:
    from ase.io import write

    source = Path(task["source"])
    output = Path(task["output"])
    original_path = task.get("original_path") or str(source)
    config_type = infer_config_type(original_path)
    neb_image = infer_neb_image(original_path)
    source_group = infer_source_group(original_path, source.name)

    try:
        frames = read_selected_frames(source, task["stride"], task["last_only"])
        converted = []
        max_force_seen = 0.0

        for frame_index, atoms in frames:
            energy, forces = finite_energy_forces(atoms)
            force_norms = __import__("numpy").linalg.norm(forces, axis=1)
            max_force = float(force_norms.max()) if len(force_norms) else 0.0
            max_force_seen = max(max_force_seen, max_force)

            if task["max_force"] is not None and max_force > task["max_force"]:
                continue

            atoms = atoms.copy()
            atoms.info["REF_energy"] = energy
            atoms.arrays["REF_forces"] = forces
            atoms.info["config_type"] = config_type
            atoms.info["source_name"] = source.name
            atoms.info["source_abs_path"] = str(source)
            atoms.info["original_abs_path"] = original_path
            atoms.info["source_group"] = source_group
            atoms.info["frame_index"] = int(frame_index)
            atoms.info["natoms"] = int(len(atoms))
            atoms.info["max_force_eVA"] = max_force
            if neb_image is not None:
                atoms.info["neb_image"] = int(neb_image)
            if task["include_stress"]:
                stress = maybe_stress(atoms)
                if stress is not None:
                    atoms.info["REF_stress"] = stress
            atoms.calc = None
            converted.append(atoms)

        if not converted:
            raise ValueError("no frames retained after filtering")

        output.parent.mkdir(parents=True, exist_ok=True)
        write(output.as_posix(), converted, format="extxyz")
        return {
            "status": "ok",
            "source": str(source),
            "output": str(output),
            "original_abs_path": original_path,
            "config_type": config_type,
            "source_group": source_group,
            "neb_image": "" if neb_image is None else neb_image,
            "frames_written": len(converted),
            "max_force_eVA": max_force_seen,
            "message": "",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "source": str(source),
            "output": str(output),
            "original_abs_path": original_path,
            "config_type": config_type,
            "source_group": source_group,
            "neb_image": "" if neb_image is None else neb_image,
            "frames_written": 0,
            "max_force_eVA": "",
            "message": repr(exc),
        }


def find_sources(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted(path.resolve() for path in iterator if path.is_file())


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "status",
        "source",
        "output",
        "original_abs_path",
        "config_type",
        "source_group",
        "neb_image",
        "frames_written",
        "max_force_eVA",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def concatenate_extxyz(outputs: list[Path], combined_output: Path) -> None:
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    with combined_output.open("wb") as write_handle:
        for output in outputs:
            with output.open("rb") as read_handle:
                shutil.copyfileobj(read_handle, write_handle, length=32 * 1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert collected OUTCAR files to per-file MACE extxyz files."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("OUTCARs"))
    parser.add_argument("--output-dir", type=Path, default=Path("MACE_extxyz"))
    parser.add_argument("--pattern", default="OUTCAR*")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest.csv from collect_outcars.py, used to restore original path metadata.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--last-only", action="store_true")
    parser.add_argument("--max-force", type=float, default=None)
    parser.add_argument(
        "--include-stress",
        action="store_true",
        help="Write REF_stress if readable. Usually avoid for slab/vacuum datasets.",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N matched files. Useful for server smoke tests.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help="Optional combined extxyz path, e.g. work/datasets/ptni_all.extxyz.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion when the target extxyz already exists.",
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    original_paths = load_manifest(args.manifest.resolve() if args.manifest else None)
    sources = find_sources(input_dir, args.pattern, args.recursive)
    if not sources:
        raise SystemExit(f"No files matching {args.pattern!r} found in {input_dir}")
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1 when provided")
        sources = sources[: args.limit]

    tasks = []
    skipped = []
    for source in sources:
        output = output_dir / output_name_for(source)
        if args.skip_existing and output.exists():
            skipped.append(
                {
                    "status": "skipped_existing",
                    "source": str(source),
                    "output": str(output),
                    "original_abs_path": original_paths.get(source.name, str(source)),
                    "config_type": infer_config_type(original_paths.get(source.name, str(source))),
                    "source_group": infer_source_group(original_paths.get(source.name, str(source)), source.name),
                    "neb_image": "",
                    "frames_written": "",
                    "max_force_eVA": "",
                    "message": "target exists",
                }
            )
            continue
        tasks.append(
            {
                "source": str(source),
                "output": str(output),
                "original_path": original_paths.get(source.name, str(source)),
                "stride": args.stride,
                "last_only": args.last_only,
                "max_force": args.max_force,
                "include_stress": args.include_stress,
            }
        )

    rows = []
    if args.jobs == 1:
        for index, task in enumerate(tasks, start=1):
            row = convert_one(task)
            rows.append(row)
            print(f"[{index}/{len(tasks)}] {row['status']} {Path(row['source']).name}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            future_to_task = {pool.submit(convert_one, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_to_task), start=1):
                row = future.result()
                rows.append(row)
                print(f"[{index}/{len(tasks)}] {row['status']} {Path(row['source']).name}")

    rows.extend(skipped)
    rows = sorted(rows, key=lambda row: row["source"])
    manifest_path = output_dir / "conversion_manifest.csv"
    write_manifest(manifest_path, rows)

    ok_outputs = [Path(row["output"]) for row in rows if row["status"] == "ok"]
    if args.combined_output is not None and ok_outputs:
        concatenate_extxyz(sorted(ok_outputs), args.combined_output.resolve())

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    failed_count = sum(1 for row in rows if row["status"] == "failed")
    skipped_count = sum(1 for row in rows if row["status"] == "skipped_existing")
    frames = sum(int(row["frames_written"]) for row in rows if row["status"] == "ok")

    print(f"Input files: {len(sources)}")
    print(f"Converted: {ok_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped existing: {skipped_count}")
    print(f"Frames written: {frames}")
    print(f"Manifest: {manifest_path}")
    if args.combined_output is not None:
        print(f"Combined extxyz: {args.combined_output.resolve()}")

    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
