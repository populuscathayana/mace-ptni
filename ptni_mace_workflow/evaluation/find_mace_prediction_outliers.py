#!/usr/bin/env python
"""Find largest energy and force errors in MACE prediction extxyz files.

This script streams prediction extxyz files and keeps only the top-N outliers
in memory. It is intended for large train/valid/test prediction files.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import re
from collections import Counter
from pathlib import Path


KEY_RE = re.compile(r'(\w+)=(".*?"|\S+)')
COMPONENTS = ("x", "y", "z")
META_KEYS = (
    "config_type",
    "source_name",
    "source_group",
    "frame_index",
    "source_abs_path",
    "original_abs_path",
)


def parse_info(comment: str) -> dict[str, str]:
    return {key: value.strip('"') for key, value in KEY_RE.findall(comment)}


def parse_properties(comment: str) -> dict[str, tuple[int, int]]:
    info = parse_info(comment)
    prop_text = info.get("Properties")
    if not prop_text:
        raise ValueError(f"comment line has no Properties= field: {comment[:200]!r}")
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


def choose_info_key(info: dict[str, str], requested: str, ref_key: str) -> str | None:
    if requested in info:
        return requested
    candidates = [
        key for key in info
        if key.lower().endswith("energy")
        and key != ref_key
        and not key.lower().startswith("ref")
    ]
    return sorted(candidates)[0] if candidates else None


def choose_array_key(props: dict[str, tuple[int, int]], requested: str, ref_key: str) -> str | None:
    if requested in props:
        return requested
    candidates = [
        key for key in props
        if key.lower().endswith("forces")
        and key != ref_key
        and not key.lower().startswith("ref")
    ]
    return sorted(candidates)[0] if candidates else None


def finite_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_xyz_array(parts: list[str], props: dict[str, tuple[int, int]], key: str) -> tuple[float, float, float]:
    start, count = props[key]
    if count != 3:
        raise ValueError(f"{key} has {count} columns, expected 3")
    return float(parts[start]), float(parts[start + 1]), float(parts[start + 2])


def species_from_parts(parts: list[str], props: dict[str, tuple[int, int]]) -> str:
    if "species" in props:
        start, _ = props["species"]
        return parts[start]
    return parts[0]


def formula_from_atom_lines(atom_lines: list[str], props: dict[str, tuple[int, int]]) -> str:
    counts = Counter(species_from_parts(line.split(), props) for line in atom_lines)
    return "".join(f"{element}{counts[element]}" for element in sorted(counts))


class TopRows:
    def __init__(self, limit: int):
        self.limit = limit
        self.heap: list[tuple[float, int, dict]] = []
        self.counter = 0

    def add(self, score: float, row: dict) -> None:
        if self.limit <= 0 or not math.isfinite(score):
            return
        item = (score, self.counter, row)
        self.counter += 1
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, item)
        elif score > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def rows_desc(self) -> list[dict]:
        rows = [row for _, _, row in sorted(self.heap, key=lambda item: item[0], reverse=True)]
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows


def metadata(info: dict[str, str]) -> dict[str, str]:
    return {key: info.get(key, "") for key in META_KEYS}


def parse_pred_arg(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use SPLIT=PATH")
    split, path = text.split("=", 1)
    split = split.strip()
    if not split:
        raise argparse.ArgumentTypeError("Split name cannot be empty")
    return split, Path(path)


def scan_file(
    split: str,
    path: Path,
    args,
    energy_top: TopRows,
    force_vector_top: TopRows,
    force_component_top: TopRows,
) -> dict:
    counters = {
        "split": split,
        "path": str(path),
        "frames": 0,
        "atoms": 0,
        "energy_frames": 0,
        "force_atoms": 0,
        "force_components": 0,
        "missing_energy": 0,
        "missing_forces": 0,
    }

    for file_frame_index, natoms, comment, atom_lines in iter_frames(path):
        info = parse_info(comment)
        props = parse_properties(comment)
        pred_energy_key = choose_info_key(info, args.pred_energy_key, args.ref_energy_key)
        pred_forces_key = choose_array_key(props, args.pred_forces_key, args.ref_forces_key)
        common = {
            "split": split,
            "file_frame_index": file_frame_index,
            "natoms": natoms,
            "formula": formula_from_atom_lines(atom_lines, props),
            **metadata(info),
        }

        ref_energy = finite_float(info.get(args.ref_energy_key))
        pred_energy = finite_float(info.get(pred_energy_key) if pred_energy_key else None)
        if ref_energy is not None and pred_energy is not None:
            ref_epa = ref_energy / natoms
            pred_epa = pred_energy / natoms
            diff_epa = pred_epa - ref_epa
            energy_top.add(
                abs(diff_epa) * 1000.0,
                {
                    **common,
                    "abs_error_mev_atom": abs(diff_epa) * 1000.0,
                    "signed_error_mev_atom": diff_epa * 1000.0,
                    "abs_total_error_ev": abs(pred_energy - ref_energy),
                    "signed_total_error_ev": pred_energy - ref_energy,
                    "dft_energy_ev": ref_energy,
                    "mace_energy_ev": pred_energy,
                    "dft_energy_ev_atom": ref_epa,
                    "mace_energy_ev_atom": pred_epa,
                    "pred_energy_key": pred_energy_key,
                },
            )
            counters["energy_frames"] += 1
        else:
            counters["missing_energy"] += 1

        if args.ref_forces_key in props and pred_forces_key:
            for atom_index, line in enumerate(atom_lines):
                parts = line.split()
                species = species_from_parts(parts, props)
                ref_vec = read_xyz_array(parts, props, args.ref_forces_key)
                pred_vec = read_xyz_array(parts, props, pred_forces_key)
                diff = tuple(pred_vec[i] - ref_vec[i] for i in range(3))
                abs_components = tuple(abs(value) for value in diff)
                vector_error = math.sqrt(sum(value * value for value in diff))
                ref_norm = math.sqrt(sum(value * value for value in ref_vec))
                pred_norm = math.sqrt(sum(value * value for value in pred_vec))
                max_i = max(range(3), key=lambda i: abs_components[i])
                force_vector_top.add(
                    vector_error * 1000.0,
                    {
                        **common,
                        "atom_index": atom_index,
                        "species": species,
                        "abs_vector_error_mev_A": vector_error * 1000.0,
                        "ref_force_norm_ev_A": ref_norm,
                        "pred_force_norm_ev_A": pred_norm,
                        "max_component": COMPONENTS[max_i],
                        "max_abs_component_error_mev_A": abs_components[max_i] * 1000.0,
                        "ref_fx_ev_A": ref_vec[0],
                        "ref_fy_ev_A": ref_vec[1],
                        "ref_fz_ev_A": ref_vec[2],
                        "pred_fx_ev_A": pred_vec[0],
                        "pred_fy_ev_A": pred_vec[1],
                        "pred_fz_ev_A": pred_vec[2],
                        "err_fx_mev_A": diff[0] * 1000.0,
                        "err_fy_mev_A": diff[1] * 1000.0,
                        "err_fz_mev_A": diff[2] * 1000.0,
                        "pred_forces_key": pred_forces_key,
                    },
                )
                for comp_i, comp in enumerate(COMPONENTS):
                    force_component_top.add(
                        abs(diff[comp_i]) * 1000.0,
                        {
                            **common,
                            "atom_index": atom_index,
                            "species": species,
                            "component": comp,
                            "abs_component_error_mev_A": abs(diff[comp_i]) * 1000.0,
                            "signed_component_error_mev_A": diff[comp_i] * 1000.0,
                            "dft_force_ev_A": ref_vec[comp_i],
                            "mace_force_ev_A": pred_vec[comp_i],
                            "pred_forces_key": pred_forces_key,
                        },
                    )
                counters["force_atoms"] += 1
                counters["force_components"] += 3
        else:
            counters["missing_forces"] += 1

        counters["frames"] += 1
        counters["atoms"] += natoms
        if args.progress and counters["frames"] % args.progress == 0:
            print(f"{split}: parsed {counters['frames']} frames")

    return counters


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    first_keys = list(rows[0].keys())
    extra_keys = sorted({key for row in rows for key in row.keys()} - set(first_keys))
    fieldnames = first_keys + extra_keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, counters: list[dict], outputs: dict[str, Path], top_n: int) -> None:
    lines = [
        "# MACE Prediction Outlier Report",
        "",
        f"Top N per category: {top_n}",
        "",
        "## Inputs",
        "",
        "| split | frames | atoms | energy frames | force atoms | missing energy | missing forces | path |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in counters:
        lines.append(
            "| {split} | {frames} | {atoms} | {energy_frames} | {force_atoms} | "
            "{missing_energy} | {missing_forces} | `{path}` |".format(**row)
        )
    lines.extend(["", "## Outputs", ""])
    for label, out_path in outputs.items():
        lines.append(f"- {label}: `{out_path}`")
    lines.extend(
        [
            "",
            "## Error Definitions",
            "",
            "- Energy outliers are ranked by `abs_error_mev_atom = abs(MACE_energy/natoms - REF_energy/natoms) * 1000`.",
            "- Force vector outliers are ranked by `abs_vector_error_mev_A = ||MACE_forces - REF_forces|| * 1000` for each atom.",
            "- Force component outliers are ranked by `abs_component_error_mev_A = abs(MACE_Fi - REF_Fi) * 1000` for each atom component.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Find largest MACE prediction energy/force errors in extxyz files.")
    parser.add_argument("--pred", action="append", type=parse_pred_arg, required=True, help="Prediction file as SPLIT=PATH.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for outlier CSV/Markdown files.")
    parser.add_argument("--top-n", type=int, default=200, help="Number of largest outliers to keep for each category.")
    parser.add_argument("--ref-energy-key", default="REF_energy")
    parser.add_argument("--pred-energy-key", default="MACE_energy")
    parser.add_argument("--ref-forces-key", default="REF_forces")
    parser.add_argument("--pred-forces-key", default="MACE_forces")
    parser.add_argument("--progress", type=int, default=1000, help="Print progress every N frames; use 0 to disable.")
    args = parser.parse_args()

    energy_top = TopRows(args.top_n)
    force_vector_top = TopRows(args.top_n)
    force_component_top = TopRows(args.top_n)
    counters = []

    for split, path in args.pred:
        resolved = path.resolve()
        print(f"Scanning {split}: {resolved}")
        counters.append(scan_file(split, resolved, args, energy_top, force_vector_top, force_component_top))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "energy_outliers": args.out_dir / "energy_outliers.csv",
        "force_vector_outliers": args.out_dir / "force_vector_outliers.csv",
        "force_component_outliers": args.out_dir / "force_component_outliers.csv",
        "summary": args.out_dir / "outlier_summary.md",
    }
    write_csv(outputs["energy_outliers"], energy_top.rows_desc())
    write_csv(outputs["force_vector_outliers"], force_vector_top.rows_desc())
    write_csv(outputs["force_component_outliers"], force_component_top.rows_desc())
    write_summary(outputs["summary"], counters, outputs, args.top_n)

    for label, out_path in outputs.items():
        print(f"{label}: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
