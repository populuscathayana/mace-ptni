#!/usr/bin/env python
"""Diagnose whether NP NEB triplets look like single-, double-, or permuted-atom events.

This script follows the idea of checkfs2.py: for identical species, compare the
original atom-index mapping with a species-preserving Hungarian assignment
between DFT 00 and DFT 02. It is meant as a diagnostic only; it does not rewrite
any input structures.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required for Hungarian matching. Install scipy in the active environment.") from exc


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def get_int_info(atoms: Any, key: str) -> int | None:
    value = atoms.info.get(key)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def apply_pbc_mode(atoms: Any, mode: str) -> None:
    if mode == "true":
        atoms.pbc = (True, True, True)
    elif mode == "false":
        atoms.pbc = (False, False, False)


def prepare_geometry(atoms: Any, wrap_scaled: bool, pbc_mode: str) -> Any:
    prepared = atoms.copy()
    if wrap_scaled:
        scaled = prepared.get_scaled_positions(wrap=False)
        prepared.set_scaled_positions(np.mod(scaled, 1.0))
    apply_pbc_mode(prepared, pbc_mode)
    return prepared


def frac_delta_pbc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = b - a
    return delta - np.round(delta)


def pairwise_cost(a: Any, b: Any, idx_a: np.ndarray, idx_b: np.ndarray, use_pbc: bool) -> np.ndarray:
    if use_pbc:
        fa = a.get_scaled_positions(wrap=False)[idx_a]
        fb = b.get_scaled_positions(wrap=False)[idx_b]
        cell = a.cell.array
        cost = np.zeros((len(idx_a), len(idx_b)), dtype=float)
        for row, frac in enumerate(fa):
            dfrac = frac_delta_pbc(frac[None, :], fb)
            dcart = dfrac @ cell
            cost[row, :] = np.sum(dcart * dcart, axis=1)
        return cost
    pa = a.get_positions()[idx_a]
    pb = b.get_positions()[idx_b]
    diff = pa[:, None, :] - pb[None, :, :]
    return np.sum(diff * diff, axis=2)


def displacement_vectors(a: Any, b: Any, mapping: np.ndarray, use_pbc: bool) -> np.ndarray:
    if use_pbc:
        fa = a.get_scaled_positions(wrap=False)
        fb = b.get_scaled_positions(wrap=False)[mapping]
        return frac_delta_pbc(fa, fb) @ a.cell.array
    return b.get_positions()[mapping] - a.get_positions()


def species_preserving_mapping(a: Any, b: Any, use_pbc: bool) -> np.ndarray:
    if len(a) != len(b):
        raise ValueError("atom counts differ")
    symbols_a = np.array(a.get_chemical_symbols())
    symbols_b = np.array(b.get_chemical_symbols())
    if sorted(symbols_a.tolist()) != sorted(symbols_b.tolist()):
        raise ValueError("chemical compositions differ")

    mapping = np.arange(len(a), dtype=int)
    for symbol in sorted(set(symbols_a.tolist())):
        idx_a = np.where(symbols_a == symbol)[0]
        idx_b = np.where(symbols_b == symbol)[0]
        cost = pairwise_cost(a, b, idx_a, idx_b, use_pbc)
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping[idx_a[row_ind]] = idx_b[col_ind]
    return mapping


def top_displacements(atoms: Any, distances: np.ndarray, mapping: np.ndarray, top_n: int) -> str:
    symbols = atoms.get_chemical_symbols()
    order = np.argsort(-distances)[:top_n]
    return ";".join(f"{int(i)}:{symbols[int(i)]}->{int(mapping[int(i)])}:{distances[int(i)]:.4f}" for i in order)


def rmse_component(vectors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(vectors * vectors))) if vectors.size else 0.0


def atom_rmsd(vectors: np.ndarray) -> float:
    distances = np.linalg.norm(vectors, axis=1)
    return float(np.sqrt(np.mean(distances * distances))) if len(distances) else 0.0


def read_triplets(configs: Path, group_key: str, image_key: str, wrap_scaled: bool, pbc_mode: str) -> dict[str, dict[int, Any]]:
    from ase.io import iread

    groups: dict[str, dict[int, Any]] = defaultdict(dict)
    for atoms in iread(configs.as_posix(), index=":"):
        group = atoms.info.get(group_key)
        image = get_int_info(atoms, image_key)
        if not group or image not in {0, 1, 2}:
            continue
        groups[str(group)][int(image)] = prepare_geometry(atoms, wrap_scaled, pbc_mode)
    return groups


def read_flagged_groups(paths: list[Path], args: argparse.Namespace) -> set[str]:
    selected: set[str] = set()
    for path in paths:
        if not path or not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                group = row.get("neb_group")
                if not group:
                    continue
                flags = (row.get("audit_flags") or "").strip()
                abnormal_flag = flags and flags != "ok_or_review"
                abnormal_force = safe_float(row.get("neb_max_force_path_eVA")) >= args.force_warn
                abnormal_neb = safe_float(row.get("abs_error_neb_forward_barrier_eV")) >= args.neb_error_warn
                abnormal_disp = safe_float(row.get("max_inner_displacement_A")) >= args.displacement_warn
                abnormal_rmsd = max(
                    safe_float(row.get("max_atom_rmsd_is_to_dft00_aligned_A")),
                    safe_float(row.get("max_atom_rmsd_ts_to_dft01_aligned_A")),
                    safe_float(row.get("max_atom_rmsd_fs_to_dft02_aligned_A")),
                ) >= args.max_atom_rmsd_warn
                if abnormal_flag or abnormal_force or abnormal_neb or abnormal_disp or abnormal_rmsd:
                    selected.add(group)
    return selected


def classify_event(active_count: int, permutation_gain: float, two_leg_active: int, top1: float, top2: float) -> str:
    if active_count <= 1:
        return "single_or_no_exchange"
    if active_count == 2 and (top2 < 0.7 or (top2 > 1e-12 and top1 / top2 >= 3.0)):
        return "single_with_secondary_relaxation"
    if active_count == 2 and permutation_gain > 0.25:
        return "possible_atom_identity_swap"
    if active_count == 2:
        return "possible_double_migration"
    if two_leg_active <= 2 and permutation_gain > 0.25:
        return "multi_index_but_permutation_helpful"
    return "multi_atom_rearrangement"


def analyze_group(group: str, triplet: dict[int, Any], args: argparse.Namespace) -> dict[str, Any]:
    is_atoms = triplet[0]
    ts_atoms = triplet[1]
    fs_atoms = triplet[2]
    use_pbc = args.metric == "pbc"

    original_mapping = np.arange(len(is_atoms), dtype=int)
    matched_mapping = species_preserving_mapping(is_atoms, fs_atoms, use_pbc)

    original_vectors = displacement_vectors(is_atoms, fs_atoms, original_mapping, use_pbc)
    matched_vectors = displacement_vectors(is_atoms, fs_atoms, matched_mapping, use_pbc)
    is_ts_vectors = displacement_vectors(is_atoms, ts_atoms, original_mapping, use_pbc)
    ts_fs_vectors = displacement_vectors(ts_atoms, fs_atoms, original_mapping, use_pbc)

    original_dist = np.linalg.norm(original_vectors, axis=1)
    matched_dist = np.linalg.norm(matched_vectors, axis=1)
    is_ts_dist = np.linalg.norm(is_ts_vectors, axis=1)
    ts_fs_dist = np.linalg.norm(ts_fs_vectors, axis=1)

    original_rmse = rmse_component(original_vectors)
    matched_rmse = rmse_component(matched_vectors)
    matched_sorted = np.sort(matched_dist)[::-1]
    top1 = float(matched_sorted[0]) if len(matched_sorted) else 0.0
    top2 = float(matched_sorted[1]) if len(matched_sorted) > 1 else 0.0
    top1_top2_ratio = "" if top2 <= 1e-12 else top1 / top2
    permutation_gain = 0.0 if original_rmse <= 1e-12 else (original_rmse - matched_rmse) / original_rmse
    active_count = int(np.sum(matched_dist >= args.active_threshold))
    original_active_count = int(np.sum(original_dist >= args.active_threshold))
    two_leg_active = int(np.sum(np.maximum(is_ts_dist, ts_fs_dist) >= args.active_threshold))

    return {
        "neb_group": group,
        "natoms": len(is_atoms),
        "metric": args.metric,
        "active_threshold_A": args.active_threshold,
        "original_component_rmse_A": original_rmse,
        "matched_component_rmse_A": matched_rmse,
        "original_atom_rmsd_A": atom_rmsd(original_vectors),
        "matched_atom_rmsd_A": atom_rmsd(matched_vectors),
        "permutation_gain_fraction": permutation_gain,
        "original_active_count": original_active_count,
        "matched_active_count": active_count,
        "two_leg_active_count_same_index": two_leg_active,
        "matched_top1_displacement_A": top1,
        "matched_top2_displacement_A": top2,
        "matched_top1_top2_ratio": top1_top2_ratio,
        "event_guess": classify_event(active_count, permutation_gain, two_leg_active, top1, top2),
        "original_top_displacements": top_displacements(is_atoms, original_dist, original_mapping, args.top_atoms),
        "matched_top_displacements": top_displacements(is_atoms, matched_dist, matched_mapping, args.top_atoms),
        "is_to_ts_top_displacements": top_displacements(is_atoms, is_ts_dist, original_mapping, args.top_atoms),
        "ts_to_fs_top_displacements": top_displacements(ts_atoms, ts_fs_dist, original_mapping, args.top_atoms),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
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


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# NP NEB Transition Identity Diagnostics",
        "",
        "This diagnostic compares original atom-index displacements with species-preserving Hungarian matching between DFT 00 and DFT 02.",
        "",
        "| group | guess | original active | matched active | top1 A | top2 A | top1/top2 | gain | top matched displacements |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {neb_group} | {event_guess} | {original_active_count} | {matched_active_count} | "
            "{matched_top1_displacement_A} | {matched_top2_displacement_A} | {matched_top1_top2_ratio} | "
            "{permutation_gain_fraction} | {matched_top_displacements} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=Path("mace_workspace/datasets/NP_benchmark_package/np_neb_benchmark_all.extxyz"))
    parser.add_argument("--summary-csv", type=Path, action="append", default=[])
    parser.add_argument("--audit-csv", type=Path, action="append", default=[])
    parser.add_argument("--group", action="append", default=[], help="Regex for groups to include. Can be repeated.")
    parser.add_argument("--all-groups", action="store_true", help="Analyze all complete triplets instead of abnormal/selected groups.")
    parser.add_argument("--out-dir", type=Path, default=Path("mace_workspace/runs/benchmarks/np_transition_identity/manual"))
    parser.add_argument("--group-key", default="neb_group")
    parser.add_argument("--image-key", default="neb_image")
    parser.add_argument("--metric", choices=["no_pbc", "pbc"], default="no_pbc")
    parser.add_argument("--pbc", choices=["from-input", "true", "false"], default="false")
    parser.add_argument("--no-wrap-scaled", dest="wrap_scaled", action="store_false")
    parser.set_defaults(wrap_scaled=True)
    parser.add_argument("--active-threshold", type=float, default=0.8)
    parser.add_argument("--top-atoms", type=int, default=6)
    parser.add_argument("--force-warn", type=float, default=1.0)
    parser.add_argument("--neb-error-warn", type=float, default=0.5)
    parser.add_argument("--displacement-warn", type=float, default=5.0)
    parser.add_argument("--max-atom-rmsd-warn", type=float, default=1.0)
    parser.add_argument("--max-groups", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = read_triplets(args.configs.resolve(), args.group_key, args.image_key, args.wrap_scaled, args.pbc)
    complete_groups = {group for group, by_image in groups.items() if set(by_image) == {0, 1, 2}}

    selected: set[str] = set()
    if args.all_groups:
        selected = set(complete_groups)
    selected |= read_flagged_groups(args.summary_csv + args.audit_csv, args)
    for pattern in args.group:
        regex = re.compile(pattern)
        selected |= {group for group in complete_groups if regex.search(group)}
    selected &= complete_groups

    if not selected:
        raise SystemExit("No complete groups selected. Use --all-groups, --group, --summary-csv, or --audit-csv.")

    selected_list = sorted(selected)
    if args.max_groups is not None:
        selected_list = selected_list[: args.max_groups]

    rows = [analyze_group(group, groups[group], args) for group in selected_list]
    rows.sort(key=lambda row: (str(row["event_guess"]), -float(row["permutation_gain_fraction"]), -int(row["matched_active_count"])))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "transition_identity_diagnostics.csv"
    md_path = args.out_dir / "transition_identity_diagnostics.md"
    write_csv(csv_path, rows)
    write_md(md_path, rows)

    print(f"Groups analyzed: {len(rows)}")
    for row in rows[: min(12, len(rows))]:
        print(
            "{neb_group}: {event_guess}, original_active={original_active_count}, "
            "matched_active={matched_active_count}, gain={permutation_gain_fraction:.3f}, top={matched_top_displacements}".format(
                **row
            )
        )
    print(f"CSV: {csv_path.resolve()}")
    print(f"Markdown: {md_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
