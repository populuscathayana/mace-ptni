#!/usr/bin/env python
"""Score NP NEB triplets from a MACE-predicted extxyz file.

The NP benchmark package contains one final-frame structure per OUTCAR. For
triplets named 00/01/02, this script compares MACE single-point energies with
DFT reference energies and reports:

- Per-structure energy and force errors.
- Per-group 00/01/02 energy table.
- Forward barrier: E(01) - E(00).
- Reverse barrier: E(01) - E(02).
- Lower-endpoint barrier: E(01) - min(E(00), E(02)).
- Reaction energy: E(02) - E(00).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


KEY_RE = re.compile(r'(\w+)=(".*?"|\S+)')


def parse_info(comment: str) -> dict[str, str]:
    return {key: value.strip('"') for key, value in KEY_RE.findall(comment)}


def parse_properties(comment: str) -> dict[str, tuple[int, int]]:
    info = parse_info(comment)
    prop_text = info.get("Properties")
    if not prop_text:
        raise ValueError("comment line has no Properties= field")
    tokens = prop_text.split(":")
    props = {}
    column = 0
    for i in range(0, len(tokens) - 2, 3):
        name = tokens[i]
        try:
            count = int(tokens[i + 2])
        except ValueError as exc:
            raise ValueError(f"invalid Properties count near {tokens[i:i+3]}") from exc
        props[name] = (column, count)
        column += count
    return props


def iter_frames(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        frame_index = 0
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            if not natoms_line.strip():
                continue
            try:
                natoms = int(natoms_line.strip())
            except ValueError as exc:
                raise ValueError(f"frame {frame_index}: invalid natoms line {natoms_line!r}") from exc
            comment = handle.readline()
            if not comment:
                raise ValueError(f"frame {frame_index}: missing comment line")
            atom_lines = [handle.readline() for _ in range(natoms)]
            if len(atom_lines) != natoms or any(line == "" for line in atom_lines):
                raise ValueError(f"frame {frame_index}: incomplete atom block")
            yield frame_index, natoms, comment.strip(), atom_lines
            frame_index += 1


def float_info(info: dict[str, str], key: str) -> float | None:
    value = info.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def int_info(info: dict[str, str], key: str) -> int | None:
    value = info.get(key)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def choose_energy_key(info: dict[str, str], requested: str, ref_key: str) -> str | None:
    if requested in info:
        return requested
    candidates = [
        key
        for key in info
        if key.lower().endswith("energy")
        and key != ref_key
        and not key.lower().startswith("ref")
    ]
    return sorted(candidates)[0] if candidates else None


def choose_force_key(props: dict[str, tuple[int, int]], requested: str, ref_key: str) -> str | None:
    if requested in props:
        return requested
    candidates = [
        key
        for key in props
        if key.lower().endswith("forces")
        and key != ref_key
        and not key.lower().startswith("ref")
    ]
    return sorted(candidates)[0] if candidates else None


def vector_triplets(atom_lines: list[str], props: dict[str, tuple[int, int]], key: str) -> list[tuple[float, float, float]]:
    start, count = props[key]
    if count != 3:
        raise ValueError(f"{key} has count {count}, expected 3")
    vectors = []
    for line in atom_lines:
        parts = line.split()
        vectors.append((float(parts[start]), float(parts[start + 1]), float(parts[start + 2])))
    return vectors


def formula_from_atom_lines(atom_lines: list[str]) -> str:
    counts = Counter(line.split()[0] for line in atom_lines if line.split())
    preferred = ["Pt", "Ni"]
    parts = []
    used = set()
    for symbol in preferred:
        if symbol in counts:
            count = counts[symbol]
            parts.append(symbol if count == 1 else f"{symbol}{count}")
            used.add(symbol)
    for symbol in sorted(counts):
        if symbol in used:
            continue
        count = counts[symbol]
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def force_error_metrics(
    atom_lines: list[str],
    props: dict[str, tuple[int, int]],
    ref_key: str,
    pred_key: str | None,
) -> dict[str, Any]:
    if ref_key not in props or pred_key is None or pred_key not in props:
        return {
            "force_component_mae_mev_A": "",
            "force_component_rmse_mev_A": "",
            "force_vector_mae_mev_A": "",
            "force_vector_rmse_mev_A": "",
            "force_vector_max_mev_A": "",
            "force_component_count": 0,
            "force_vector_count": 0,
        }

    ref = vector_triplets(atom_lines, props, ref_key)
    pred = vector_triplets(atom_lines, props, pred_key)
    comp_abs = 0.0
    comp_sq = 0.0
    comp_count = 0
    vec_abs = 0.0
    vec_sq = 0.0
    vec_max = 0.0
    vec_count = 0
    for r, p in zip(ref, pred):
        diff = (p[0] - r[0], p[1] - r[1], p[2] - r[2])
        for value in diff:
            comp_abs += abs(value)
            comp_sq += value * value
            comp_count += 1
        norm = math.sqrt(sum(value * value for value in diff))
        vec_abs += norm
        vec_sq += norm * norm
        vec_max = max(vec_max, norm)
        vec_count += 1

    return {
        "force_component_mae_mev_A": 1000.0 * comp_abs / comp_count if comp_count else "",
        "force_component_rmse_mev_A": 1000.0 * math.sqrt(comp_sq / comp_count) if comp_count else "",
        "force_vector_mae_mev_A": 1000.0 * vec_abs / vec_count if vec_count else "",
        "force_vector_rmse_mev_A": 1000.0 * math.sqrt(vec_sq / vec_count) if vec_count else "",
        "force_vector_max_mev_A": 1000.0 * vec_max if vec_count else "",
        "force_component_count": comp_count,
        "force_vector_count": vec_count,
    }


def parse_predicted_extxyz(
    path: Path,
    ref_energy_key: str,
    pred_energy_key: str,
    ref_forces_key: str,
    pred_forces_key: str,
    group_key: str,
    image_key: str,
) -> list[dict[str, Any]]:
    rows = []
    used_pred_energy_key = None
    used_pred_forces_key = None
    for file_frame_index, natoms, comment, atom_lines in iter_frames(path):
        info = parse_info(comment)
        props = parse_properties(comment)
        energy_key = choose_energy_key(info, pred_energy_key, ref_energy_key)
        forces_key = choose_force_key(props, pred_forces_key, ref_forces_key)
        if used_pred_energy_key is None:
            used_pred_energy_key = energy_key
        if used_pred_forces_key is None:
            used_pred_forces_key = forces_key

        ref_energy = float_info(info, ref_energy_key)
        pred_energy = float_info(info, energy_key) if energy_key else None
        energy_error = None
        energy_error_mev_atom = None
        if ref_energy is not None and pred_energy is not None:
            energy_error = pred_energy - ref_energy
            energy_error_mev_atom = 1000.0 * energy_error / natoms

        force_metrics = force_error_metrics(atom_lines, props, ref_forces_key, forces_key)
        row = {
            "file_frame_index": file_frame_index,
            "natoms": natoms,
            "formula": formula_from_atom_lines(atom_lines),
            "config_type": info.get("config_type", "Default"),
            "source_name": info.get("source_name", ""),
            "source_group": info.get("source_group", ""),
            "neb_group": info.get(group_key, ""),
            "neb_image": int_info(info, image_key),
            "neb_role": info.get("neb_role", ""),
            "neb_group_complete": int_info(info, "neb_group_complete"),
            "ref_energy_eV": ref_energy,
            "pred_energy_eV": pred_energy,
            "energy_error_eV": energy_error,
            "energy_error_mev_atom": energy_error_mev_atom,
            "pred_energy_key": energy_key or "",
            "pred_forces_key": forces_key or "",
        }
        row.update(force_metrics)
        rows.append(row)

    if rows:
        for row in rows:
            if not row["pred_energy_key"] and used_pred_energy_key:
                row["pred_energy_key"] = used_pred_energy_key
            if not row["pred_forces_key"] and used_pred_forces_key:
                row["pred_forces_key"] = used_pred_forces_key
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def error_stats(values: list[float]) -> dict[str, float | str]:
    if not values:
        return {"mae": "", "rmse": "", "mean": "", "max_abs": ""}
    abs_values = [abs(value) for value in values]
    return {
        "mae": sum(abs_values) / len(values),
        "rmse": math.sqrt(sum(value * value for value in values) / len(values)),
        "mean": sum(values) / len(values),
        "max_abs": max(abs_values),
    }


def build_group_rows(structure_rows: list[dict[str, Any]], complete_only: bool) -> list[dict[str, Any]]:
    groups: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in structure_rows:
        group = row.get("neb_group")
        image = row.get("neb_image")
        if not group or image not in {0, 1, 2}:
            continue
        groups[str(group)][int(image)] = row

    rows = []
    for group, by_image in sorted(groups.items()):
        present = sorted(by_image)
        complete = set(present) == {0, 1, 2}
        if complete_only and not complete:
            continue

        row: dict[str, Any] = {
            "neb_group": group,
            "images_present": ";".join(f"{image:02d}" for image in present),
            "complete_00_01_02": complete,
        }
        for image in (0, 1, 2):
            image_row = by_image.get(image)
            for prefix, key in (("ref", "ref_energy_eV"), ("pred", "pred_energy_eV")):
                row[f"{prefix}_E{image:02d}_eV"] = image_row.get(key) if image_row else ""
            row[f"source_{image:02d}"] = image_row.get("source_name", "") if image_row else ""
            row[f"natoms_{image:02d}"] = image_row.get("natoms", "") if image_row else ""
            row[f"energy_error_{image:02d}_mev_atom"] = image_row.get("energy_error_mev_atom", "") if image_row else ""

        if complete and all(finite(row.get(f"ref_E{image:02d}_eV")) for image in (0, 1, 2)) and all(
            finite(row.get(f"pred_E{image:02d}_eV")) for image in (0, 1, 2)
        ):
            ref0 = float(row["ref_E00_eV"])
            ref1 = float(row["ref_E01_eV"])
            ref2 = float(row["ref_E02_eV"])
            pred0 = float(row["pred_E00_eV"])
            pred1 = float(row["pred_E01_eV"])
            pred2 = float(row["pred_E02_eV"])
            metrics = {
                "forward_barrier_eV": (ref1 - ref0, pred1 - pred0),
                "reverse_barrier_eV": (ref1 - ref2, pred1 - pred2),
                "lower_endpoint_barrier_eV": (ref1 - min(ref0, ref2), pred1 - min(pred0, pred2)),
                "reaction_energy_eV": (ref2 - ref0, pred2 - pred0),
            }
            for name, (ref_value, pred_value) in metrics.items():
                row[f"ref_{name}"] = ref_value
                row[f"pred_{name}"] = pred_value
                row[f"error_{name}"] = pred_value - ref_value
                row[f"abs_error_{name}"] = abs(pred_value - ref_value)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_summary(path: Path, structure_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]], pred_path: Path) -> None:
    energy_errors = [float(row["energy_error_mev_atom"]) for row in structure_rows if finite(row.get("energy_error_mev_atom"))]
    force_comp_errors = []
    force_vec_errors = []
    for row in structure_rows:
        if finite(row.get("force_component_rmse_mev_A")):
            force_comp_errors.append(float(row["force_component_rmse_mev_A"]))
        if finite(row.get("force_vector_rmse_mev_A")):
            force_vec_errors.append(float(row["force_vector_rmse_mev_A"]))

    metric_names = [
        "forward_barrier_eV",
        "reverse_barrier_eV",
        "lower_endpoint_barrier_eV",
        "reaction_energy_eV",
    ]
    metric_stats = {}
    for name in metric_names:
        errors = [1000.0 * float(row[f"error_{name}"]) for row in group_rows if finite(row.get(f"error_{name}"))]
        metric_stats[name] = error_stats(errors)

    complete_groups = [row for row in group_rows if row.get("complete_00_01_02") is True]
    top_forward = sorted(
        [row for row in group_rows if finite(row.get("abs_error_forward_barrier_eV"))],
        key=lambda row: float(row["abs_error_forward_barrier_eV"]),
        reverse=True,
    )[:10]

    e_stats = error_stats(energy_errors)
    lines = [
        "# NP NEB Single-Point Benchmark",
        "",
        f"- Prediction file: `{pred_path.resolve()}`",
        f"- Structures scored: {len(structure_rows)}",
        f"- NEB groups scored: {len(group_rows)}",
        f"- Complete 00/01/02 groups: {len(complete_groups)}",
        "",
        "## Structure-Level Errors",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| Energy MAE | {fmt(e_stats['mae'])} meV/atom |",
        f"| Energy RMSE | {fmt(e_stats['rmse'])} meV/atom |",
        f"| Energy max abs | {fmt(e_stats['max_abs'])} meV/atom |",
        f"| Mean per-structure force component RMSE | {fmt(sum(force_comp_errors) / len(force_comp_errors) if force_comp_errors else '')} meV/A |",
        f"| Mean per-structure force vector RMSE | {fmt(sum(force_vec_errors) / len(force_vec_errors) if force_vec_errors else '')} meV/A |",
        "",
        "## NEB Group Relative-Energy Errors",
        "",
        "| quantity | MAE meV | RMSE meV | mean signed meV | max abs meV |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "forward_barrier_eV": "E01 - E00",
        "reverse_barrier_eV": "E01 - E02",
        "lower_endpoint_barrier_eV": "E01 - min(E00,E02)",
        "reaction_energy_eV": "E02 - E00",
    }
    for name in metric_names:
        stats = metric_stats[name]
        lines.append(
            f"| {labels[name]} | {fmt(stats['mae'])} | {fmt(stats['rmse'])} | "
            f"{fmt(stats['mean'])} | {fmt(stats['max_abs'])} |"
        )

    lines.extend(
        [
            "",
            "## Largest Forward-Barrier Outliers",
            "",
            "| rank | neb_group | ref eV | pred eV | error meV |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(top_forward, start=1):
        lines.append(
            f"| {rank} | `{row['neb_group']}` | {fmt(row.get('ref_forward_barrier_eV'), 6)} | "
            f"{fmt(row.get('pred_forward_barrier_eV'), 6)} | {fmt(1000.0 * float(row['error_forward_barrier_eV']))} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score NP NEB triplet single-point MACE predictions.")
    parser.add_argument("--pred", type=Path, required=True, help="Predicted extxyz containing REF_* and MACE_* keys.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--ref-energy-key", default="REF_energy")
    parser.add_argument("--pred-energy-key", default="MACE_energy")
    parser.add_argument("--ref-forces-key", default="REF_forces")
    parser.add_argument("--pred-forces-key", default="MACE_forces")
    parser.add_argument("--group-key", default="neb_group")
    parser.add_argument("--image-key", default="neb_image")
    parser.add_argument(
        "--include-incomplete-groups",
        action="store_true",
        help="Also write rows for groups missing one of 00/01/02. Relative metrics are blank.",
    )
    args = parser.parse_args()

    pred = args.pred.resolve()
    if not pred.is_file():
        raise SystemExit(f"Prediction file does not exist: {pred}")
    name = args.name or pred.stem
    out_dir = args.out_dir.resolve()

    structure_rows = parse_predicted_extxyz(
        pred,
        ref_energy_key=args.ref_energy_key,
        pred_energy_key=args.pred_energy_key,
        ref_forces_key=args.ref_forces_key,
        pred_forces_key=args.pred_forces_key,
        group_key=args.group_key,
        image_key=args.image_key,
    )
    group_rows = build_group_rows(structure_rows, complete_only=not args.include_incomplete_groups)

    per_structure_csv = out_dir / f"{name}_per_structure_errors.csv"
    group_csv = out_dir / f"{name}_neb_group_barriers.csv"
    summary_md = out_dir / f"{name}_np_neb_summary.md"
    write_csv(per_structure_csv, structure_rows)
    write_csv(group_csv, group_rows)
    write_summary(summary_md, structure_rows, group_rows, pred)

    print(f"Structures scored: {len(structure_rows)}")
    print(f"NEB group rows: {len(group_rows)}")
    print(f"Per-structure CSV: {per_structure_csv}")
    print(f"Group barrier CSV: {group_csv}")
    print(f"Summary: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
