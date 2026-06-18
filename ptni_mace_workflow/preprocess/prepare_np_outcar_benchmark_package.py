#!/usr/bin/env python
"""Prepare NP OUTCAR files as a labeled MACE benchmark package.

This script is intended for the NP holdout/extrapolation benchmark:

1. Find OUTCAR-like files.
2. Keep only allowed elements, and by default require Pt_pv_GW for Pt files.
3. Deduplicate by size first, then SHA-256 for same-size files.
4. Recognize flat NEB triplets named like *_00_OUTCAR.OUTCAR,
   *_01_OUTCAR.OUTCAR, *_02_OUTCAR.OUTCAR.
5. Convert the retained OUTCARs to MACE-ready extxyz with REF_energy and
   REF_forces. By default only the last OUTCAR frame is exported.
6. Write manifests and a small tar.gz package containing extxyz + CSV metadata.

Original OUTCAR files are never deleted by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


NEB_NAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<image>0?[012])_OUTCAR(?:\.OUTCAR)?$",
    re.IGNORECASE,
)


def ascii_safe(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("ascii", errors="backslashreplace").decode("ascii")
    return value


def parse_allowed_elements(text: str) -> set[str]:
    allowed = {item.strip() for item in text.split(",") if item.strip()}
    if not allowed:
        raise SystemExit("--allowed-elements cannot be empty")
    bad = [item for item in allowed if not re.fullmatch(r"[A-Z][a-z]?", item)]
    if bad:
        raise SystemExit(f"Invalid element symbols in --allowed-elements: {bad}")
    return allowed


def find_outcars(root: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(path.resolve() for path in iterator if path.is_file())


def extract_potcar_labels_from_line(line: str) -> list[str]:
    if "POTCAR:" in line:
        text = line.split("POTCAR:", 1)[1]
    elif "TITEL" in line and "PAW" in line:
        text = line.split("=", 1)[-1]
    else:
        return []

    labels = []
    for raw in text.replace(";", " ").replace(",", " ").split():
        token = raw.strip()
        if not token:
            continue
        if token.startswith("PAW"):
            continue
        if re.fullmatch(r"[A-Z][a-z]?(?:_[A-Za-z0-9]+)*", token):
            labels.append(token)
    return labels


def label_to_symbol(label: str) -> str | None:
    match = re.match(r"^([A-Z][a-z]?)", label)
    return match.group(1) if match else None


def ordered_unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def scan_potcar_labels(path: Path, max_lines: int, max_context: int) -> dict[str, Any]:
    labels = []
    evidence = []
    lines_scanned = 0
    stopped_at_line_limit = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_lines > 0 and line_number > max_lines:
                stopped_at_line_limit = True
                break
            lines_scanned = line_number
            line_labels = extract_potcar_labels_from_line(line)
            if line_labels:
                labels.extend(line_labels)
                if len(evidence) < max_context:
                    evidence.append(line.strip())

    labels = ordered_unique(labels)
    symbols = ordered_unique([symbol for label in labels if (symbol := label_to_symbol(label))])
    return {
        "potcar_labels": labels,
        "potcar_symbols": symbols,
        "potcar_evidence": evidence,
        "lines_scanned": lines_scanned,
        "stopped_at_line_limit": stopped_at_line_limit,
    }


def sha256_file(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_name_from_path(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    text = rel.as_posix()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._")
    return text or path.name


def extxyz_name_from_safe_outcar_name(safe_name: str) -> str:
    name = safe_name
    if name.lower().endswith(".outcar"):
        name = name[: -len(".outcar")]
    if not name:
        name = "OUTCAR"
    return f"{name}.extxyz"


def infer_neb_info(root: Path, path: Path) -> dict[str, Any]:
    match = NEB_NAME_RE.match(path.name)
    if not match:
        return {
            "neb_group": "",
            "neb_prefix": "",
            "neb_image": "",
            "neb_role": "",
            "is_neb_triplet_name": False,
        }

    prefix = match.group("prefix")
    image = int(match.group("image"))
    try:
        rel_parent = path.parent.relative_to(root).as_posix()
    except ValueError:
        rel_parent = path.parent.as_posix()
    group = prefix if rel_parent in ("", ".") else f"{rel_parent}/{prefix}"
    role = {0: "initial", 1: "transition", 2: "final"}[image]
    return {
        "neb_group": group,
        "neb_prefix": prefix,
        "neb_image": image,
        "neb_role": role,
        "is_neb_triplet_name": True,
    }


def config_type_for_record(record: dict[str, Any]) -> str:
    if record.get("is_neb_triplet_name"):
        if record.get("neb_image") == 1:
            return "np_neb_ts"
        return "np_neb_endpoint"
    return "np_singlepoint"


def make_initial_records(
    root: Path,
    files: list[Path],
    allowed_elements: set[str],
    require_pt_pv_gw: bool,
    max_lines: int,
    max_context: int,
) -> list[dict[str, Any]]:
    rows = []
    for index, path in enumerate(files, start=1):
        stat = path.stat()
        scan = scan_potcar_labels(path, max_lines=max_lines, max_context=max_context)
        labels = set(scan["potcar_labels"])
        symbols = set(scan["potcar_symbols"])

        reject_reason = ""
        if not symbols:
            reject_reason = "missing_potcar_evidence"
        elif not symbols.issubset(allowed_elements):
            reject_reason = "disallowed_elements"
        elif require_pt_pv_gw and "Pt" in symbols and "Pt_pv_GW" not in labels:
            reject_reason = "pt_without_pt_pv_gw"

        neb_info = infer_neb_info(root, path)
        safe_name = safe_name_from_path(root, path)
        rows.append(
            {
                "index": index,
                "path": path,
                "input_path": str(path),
                "input_name": path.name,
                "safe_name": safe_name,
                "size_bytes": stat.st_size,
                "potcar_labels": ";".join(scan["potcar_labels"]),
                "potcar_symbols": ";".join(scan["potcar_symbols"]),
                "potcar_evidence": " || ".join(scan["potcar_evidence"]),
                "lines_scanned": scan["lines_scanned"],
                "stopped_at_line_limit": scan["stopped_at_line_limit"],
                "element_filter_status": "reject" if reject_reason else "keep",
                "reject_reason": reject_reason,
                "sha256": "",
                "duplicate_of": "",
                "keep_after_dedupe": False,
                "final_keep": False,
                "clean_outcar_path": "",
                "extxyz_path": "",
                "config_type": "",
                "neb_group": neb_info["neb_group"],
                "neb_prefix": neb_info["neb_prefix"],
                "neb_image": neb_info["neb_image"],
                "neb_role": neb_info["neb_role"],
                "neb_group_complete": False,
                "is_neb_triplet_name": neb_info["is_neb_triplet_name"],
            }
        )
        if index % 50 == 0 or index == len(files):
            print(f"Scanned POTCAR labels {index}/{len(files)}")
    return rows


def apply_size_hash_dedupe(records: list[dict[str, Any]], chunk_size: int) -> None:
    kept_candidates = [row for row in records if not row["reject_reason"]]
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in kept_candidates:
        by_size[int(row["size_bytes"])].append(row)

    for size, rows in sorted(by_size.items()):
        if len(rows) == 1:
            rows[0]["keep_after_dedupe"] = True
            continue

        by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sorted(rows, key=lambda item: item["input_path"]):
            digest = sha256_file(row["path"], chunk_size)
            row["sha256"] = digest
            by_hash[digest].append(row)

        for digest, digest_rows in sorted(by_hash.items()):
            keeper = sorted(digest_rows, key=lambda item: item["input_path"])[0]
            keeper["keep_after_dedupe"] = True
            for duplicate in sorted(digest_rows, key=lambda item: item["input_path"])[1:]:
                duplicate["duplicate_of"] = keeper["input_path"]
                duplicate["reject_reason"] = "duplicate_content"


def finalize_neb_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_records = [row for row in records if row["keep_after_dedupe"] and not row["reject_reason"]]
    groups: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in final_records:
        if row["neb_group"] and row["neb_image"] != "":
            groups[row["neb_group"]][int(row["neb_image"])] = row

    group_rows = []
    for group, image_rows in sorted(groups.items()):
        present = sorted(image_rows)
        complete = set(present) == {0, 1, 2}
        for row in image_rows.values():
            row["neb_group_complete"] = complete
        group_rows.append(
            {
                "neb_group": group,
                "images_present": ";".join(f"{image:02d}" for image in present),
                "complete_00_01_02": complete,
                "image_00_outcar": image_rows.get(0, {}).get("input_path", ""),
                "image_01_outcar": image_rows.get(1, {}).get("input_path", ""),
                "image_02_outcar": image_rows.get(2, {}).get("input_path", ""),
                "image_00_extxyz": "",
                "image_01_extxyz": "",
                "image_02_extxyz": "",
            }
        )

    for row in final_records:
        row["final_keep"] = True
        row["config_type"] = config_type_for_record(row)
    return group_rows


def copy_clean_outcars(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in records:
        if not row["final_keep"]:
            continue
        target = output_dir / row["safe_name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], target)
        row["clean_outcar_path"] = str(target)


def finite_energy_forces(atoms: Any) -> tuple[float, Any]:
    import numpy as np

    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if forces.shape != (len(atoms), 3):
        raise ValueError(f"unexpected forces shape {forces.shape}")
    if not math.isfinite(energy) or not np.isfinite(forces).all():
        raise ValueError("non-finite energy or forces")
    return energy, forces


def read_selected_frames(path: Path, stride: int, last_only: bool) -> list[tuple[int, Any]]:
    from ase.io import iread, read

    if last_only:
        return [(-1, read(path.as_posix(), index=-1, format="vasp-out"))]

    selected = []
    for index, atoms in enumerate(iread(path.as_posix(), index=":", format="vasp-out")):
        if index % stride == 0:
            selected.append((index, atoms))
    return selected


def convert_one(task: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from ase.io import write

    source = Path(task["source"])
    output = Path(task["output"])
    last_only = bool(task["last_only"])
    stride = int(task["stride"])
    metadata = task["metadata"]
    try:
        frames = read_selected_frames(source, stride=stride, last_only=last_only)
        converted = []
        max_force_seen = 0.0
        for frame_index, atoms in frames:
            energy, forces = finite_energy_forces(atoms)
            force_norms = np.linalg.norm(forces, axis=1)
            max_force = float(force_norms.max()) if len(force_norms) else 0.0
            max_force_seen = max(max_force_seen, max_force)

            atoms = atoms.copy()
            atoms.info["REF_energy"] = energy
            atoms.arrays["REF_forces"] = forces
            atoms.info["np_benchmark"] = 1
            atoms.info["frame_index"] = int(frame_index)
            atoms.info["natoms"] = int(len(atoms))
            atoms.info["max_force_eVA"] = max_force
            for key, value in metadata.items():
                if value == "":
                    continue
                atoms.info[key] = ascii_safe(value)
            atoms.calc = None
            converted.append(atoms)

        if not converted:
            raise ValueError("no frames selected")

        output.parent.mkdir(parents=True, exist_ok=True)
        write(output.as_posix(), converted, format="extxyz")
        return {
            "status": "ok",
            "source": str(source),
            "output": str(output),
            "frames_written": len(converted),
            "max_force_eVA": max_force_seen,
            "message": "",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "source": str(source),
            "output": str(output),
            "frames_written": 0,
            "max_force_eVA": "",
            "message": repr(exc),
        }


def concatenate_extxyz(outputs: list[Path], combined_output: Path) -> None:
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    with combined_output.open("wb") as write_handle:
        for output in outputs:
            with output.open("rb") as read_handle:
                shutil.copyfileobj(read_handle, write_handle, length=32 * 1024 * 1024)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, Path):
                    value = str(value)
                serializable[key] = value
            writer.writerow(serializable)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_group_extxyz_paths(group_rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    by_group_image = {}
    for row in records:
        if row["final_keep"] and row["neb_group"] and row["neb_image"] != "":
            by_group_image[(row["neb_group"], int(row["neb_image"]))] = row.get("extxyz_path", "")
    for group_row in group_rows:
        group = group_row["neb_group"]
        for image in (0, 1, 2):
            group_row[f"image_{image:02d}_extxyz"] = by_group_image.get((group, image), "")


def create_archive(
    archive_path: Path,
    files: list[Path],
    base_dir: Path,
    extra_dirs: list[Path],
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in files:
            if path.exists():
                tar.add(path, arcname=path.relative_to(base_dir))
        for directory in extra_dirs:
            if directory.exists():
                tar.add(directory, arcname=directory.relative_to(base_dir))


def checksum_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if path.exists() and path.is_file():
            rows.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path, 32 * 1024 * 1024),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare NP OUTCAR files for a MACE extrapolation benchmark package."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("OUTCARs/NP"))
    parser.add_argument("--work-dir", type=Path, default=Path("NP_benchmark_package"))
    parser.add_argument("--pattern", default="*OUTCAR*")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--allowed-elements", default="Pt,Ni")
    parser.add_argument(
        "--allow-pt-pv",
        action="store_true",
        help="Do not require Pt_pv_GW for files containing Pt. Default requires Pt_pv_GW.",
    )
    parser.add_argument(
        "--potcar-max-lines",
        type=int,
        default=100,
        help="Read only the first N lines for POTCAR/TITEL evidence. Use 0 for full file.",
    )
    parser.add_argument("--potcar-context-lines", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N candidates.")
    parser.add_argument("--name", default="np_neb_benchmark")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Export every selected OUTCAR frame instead of only the last frame.",
    )
    parser.add_argument(
        "--copy-clean-outcars",
        action="store_true",
        help="Copy retained unique OUTCARs to <work-dir>/OUTCARs_clean.",
    )
    parser.add_argument(
        "--include-clean-outcars-in-archive",
        action="store_true",
        help="Include OUTCARs_clean in the tar.gz archive. Requires --copy-clean-outcars.",
    )
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    if args.potcar_max_lines < 0:
        raise SystemExit("--potcar-max-lines must be >= 0")
    if args.potcar_context_lines < 0:
        raise SystemExit("--potcar-context-lines must be >= 0")
    if args.chunk_mb < 1:
        raise SystemExit("--chunk-mb must be >= 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1 when provided")
    if args.include_clean_outcars_in_archive and not args.copy_clean_outcars:
        raise SystemExit("--include-clean-outcars-in-archive requires --copy-clean-outcars")

    input_dir = args.input_dir.resolve()
    work_dir = args.work_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    allowed_elements = parse_allowed_elements(args.allowed_elements)
    chunk_size = args.chunk_mb * 1024 * 1024
    extxyz_dir = work_dir / "MACE_extxyz"
    clean_dir = work_dir / "OUTCARs_clean"
    manifest_path = work_dir / "np_outcar_manifest.csv"
    conversion_manifest_path = work_dir / "np_conversion_manifest.csv"
    group_manifest_path = work_dir / "np_neb_groups.csv"
    combined_extxyz = work_dir / f"{args.name}_all.extxyz"
    summary_path = work_dir / f"{args.name}_summary.json"
    checksums_path = work_dir / f"{args.name}_checksums.csv"
    archive_path = work_dir / f"{args.name}_mace_extxyz_package.tar.gz"

    files = find_outcars(input_dir, args.pattern, args.recursive)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} found in {input_dir}")

    records = make_initial_records(
        root=input_dir,
        files=files,
        allowed_elements=allowed_elements,
        require_pt_pv_gw=not args.allow_pt_pv,
        max_lines=args.potcar_max_lines,
        max_context=args.potcar_context_lines,
    )
    apply_size_hash_dedupe(records, chunk_size=chunk_size)
    group_rows = finalize_neb_groups(records)

    final_records = [row for row in records if row["final_keep"]]
    for row in final_records:
        output = extxyz_dir / extxyz_name_from_safe_outcar_name(row["safe_name"])
        row["extxyz_path"] = str(output)

    if args.copy_clean_outcars:
        copy_clean_outcars(records, clean_dir)

    tasks = []
    for row in final_records:
        metadata = {
            "config_type": row["config_type"],
            "source_name": row["input_name"],
            "source_abs_path": row["input_path"],
            "original_abs_path": row["input_path"],
            "source_group": row["neb_group"] or Path(row["safe_name"]).stem,
            "potcar_labels": row["potcar_labels"],
            "potcar_symbols": row["potcar_symbols"],
            "neb_group": row["neb_group"],
            "neb_image": row["neb_image"],
            "neb_role": row["neb_role"],
            "neb_group_complete": int(bool(row["neb_group_complete"])),
        }
        tasks.append(
            {
                "source": row["input_path"],
                "output": row["extxyz_path"],
                "last_only": not args.all_frames,
                "stride": args.stride,
                "metadata": metadata,
            }
        )

    conversion_rows = []
    if args.jobs == 1:
        for index, task in enumerate(tasks, start=1):
            result = convert_one(task)
            conversion_rows.append(result)
            print(f"[{index}/{len(tasks)}] {result['status']} {Path(result['source']).name}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(convert_one, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                conversion_rows.append(result)
                print(f"[{index}/{len(tasks)}] {result['status']} {Path(result['source']).name}")

    conversion_by_source = {row["source"]: row for row in conversion_rows}
    ok_outputs = []
    for row in final_records:
        conversion = conversion_by_source.get(row["input_path"])
        if conversion and conversion["status"] == "ok":
            ok_outputs.append(Path(row["extxyz_path"]))

    if ok_outputs:
        concatenate_extxyz(sorted(ok_outputs), combined_extxyz)

    update_group_extxyz_paths(group_rows, records)

    outcar_fields = [
        "index",
        "input_path",
        "input_name",
        "safe_name",
        "size_bytes",
        "potcar_labels",
        "potcar_symbols",
        "potcar_evidence",
        "lines_scanned",
        "stopped_at_line_limit",
        "element_filter_status",
        "reject_reason",
        "sha256",
        "duplicate_of",
        "keep_after_dedupe",
        "final_keep",
        "clean_outcar_path",
        "extxyz_path",
        "config_type",
        "neb_group",
        "neb_prefix",
        "neb_image",
        "neb_role",
        "neb_group_complete",
        "is_neb_triplet_name",
    ]
    group_fields = [
        "neb_group",
        "images_present",
        "complete_00_01_02",
        "image_00_outcar",
        "image_01_outcar",
        "image_02_outcar",
        "image_00_extxyz",
        "image_01_extxyz",
        "image_02_extxyz",
    ]
    conversion_fields = [
        "status",
        "source",
        "output",
        "frames_written",
        "max_force_eVA",
        "message",
    ]
    write_csv(manifest_path, records, outcar_fields)
    write_csv(group_manifest_path, group_rows, group_fields)
    write_csv(conversion_manifest_path, sorted(conversion_rows, key=lambda row: row["source"]), conversion_fields)

    rejected = [row for row in records if row["reject_reason"] and row["reject_reason"] != "duplicate_content"]
    duplicates = [row for row in records if row["reject_reason"] == "duplicate_content"]
    complete_groups = [row for row in group_rows if row["complete_00_01_02"]]
    incomplete_groups = [row for row in group_rows if not row["complete_00_01_02"]]
    ok_count = sum(1 for row in conversion_rows if row["status"] == "ok")
    failed_count = sum(1 for row in conversion_rows if row["status"] != "ok")
    frames_written = sum(int(row["frames_written"]) for row in conversion_rows if row["status"] == "ok")
    timestamp = datetime.now().isoformat(timespec="seconds")

    summary = {
        "generated_at": timestamp,
        "input_dir": str(input_dir),
        "work_dir": str(work_dir),
        "candidate_files": len(files),
        "rejected_by_filter": len(rejected),
        "duplicates_removed_from_package": len(duplicates),
        "unique_retained_outcars": len(final_records),
        "complete_neb_groups": len(complete_groups),
        "incomplete_neb_groups": len(incomplete_groups),
        "converted_outcars": ok_count,
        "failed_conversions": failed_count,
        "frames_written": frames_written,
        "last_frame_only": not args.all_frames,
        "combined_extxyz": str(combined_extxyz) if combined_extxyz.exists() else "",
        "archive": str(archive_path) if not args.no_archive else "",
    }
    write_json(summary_path, summary)

    files_for_checksums = [
        manifest_path,
        group_manifest_path,
        conversion_manifest_path,
        combined_extxyz,
        summary_path,
    ]
    write_csv(checksums_path, checksum_rows(files_for_checksums), ["path", "size_bytes", "sha256"])

    archive_created = False
    if not args.no_archive:
        archive_files = [
            manifest_path,
            group_manifest_path,
            conversion_manifest_path,
            checksums_path,
            combined_extxyz,
            summary_path,
        ]
        extra_dirs = [clean_dir] if args.include_clean_outcars_in_archive else []
        create_archive(archive_path, archive_files, base_dir=work_dir, extra_dirs=extra_dirs)
        archive_created = True

    print(f"Candidate OUTCAR files: {len(files)}")
    print(f"Rejected by element/POTCAR filter: {len(rejected)}")
    print(f"Duplicate files excluded from package: {len(duplicates)}")
    print(f"Unique retained OUTCAR files: {len(final_records)}")
    print(f"Complete NEB groups: {len(complete_groups)}")
    print(f"Incomplete NEB groups: {len(incomplete_groups)}")
    print(f"Converted OUTCAR files: {ok_count}")
    print(f"Failed conversions: {failed_count}")
    print(f"Frames written: {frames_written}")
    print(f"Combined extxyz: {combined_extxyz}")
    print(f"OUTCAR manifest: {manifest_path}")
    print(f"NEB group manifest: {group_manifest_path}")
    if archive_created:
        print(f"Archive: {archive_path}")

    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
