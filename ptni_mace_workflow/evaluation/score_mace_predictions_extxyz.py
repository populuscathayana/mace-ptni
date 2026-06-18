#!/usr/bin/env python
"""Score MACE prediction extxyz files against REF_energy/REF_forces.

The parser is streaming and does not require ASE. It computes overall and
config_type-level errors for train/valid/test prediction files produced by
mace_eval_configs.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


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
        value = float(value)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def choose_force_key(props: dict[str, tuple[int, int]], requested: str, ref_key: str) -> str | None:
    if requested in props:
        return requested
    candidates = [
        key for key in props
        if key.lower().endswith("forces")
        and key != ref_key
        and not key.lower().startswith("ref")
    ]
    return sorted(candidates)[0] if candidates else None


def choose_energy_key(info: dict[str, str], requested: str, ref_key: str) -> str | None:
    if requested in info:
        return requested
    candidates = [
        key for key in info
        if key.lower().endswith("energy")
        and key != ref_key
        and not key.lower().startswith("ref")
    ]
    return sorted(candidates)[0] if candidates else None


class Accumulator:
    def __init__(self):
        self.frames = 0
        self.atoms = 0
        self.energy_abs_sum = 0.0
        self.energy_sq_sum = 0.0
        self.energy_count = 0
        self.force_abs_sum = 0.0
        self.force_sq_sum = 0.0
        self.force_component_count = 0
        self.force_vector_abs_sum = 0.0
        self.force_vector_sq_sum = 0.0
        self.force_vector_count = 0
        self.missing_energy = 0
        self.missing_forces = 0

    def add_energy(self, diff_per_atom_ev: float | None):
        if diff_per_atom_ev is None:
            self.missing_energy += 1
            return
        self.energy_abs_sum += abs(diff_per_atom_ev)
        self.energy_sq_sum += diff_per_atom_ev * diff_per_atom_ev
        self.energy_count += 1

    def add_force_components(self, diffs: list[tuple[float, float, float]] | None):
        if diffs is None:
            self.missing_forces += 1
            return
        for dx, dy, dz in diffs:
            for value in (dx, dy, dz):
                self.force_abs_sum += abs(value)
                self.force_sq_sum += value * value
                self.force_component_count += 1
            norm = math.sqrt(dx * dx + dy * dy + dz * dz)
            self.force_vector_abs_sum += norm
            self.force_vector_sq_sum += norm * norm
            self.force_vector_count += 1

    def row(self, split: str, group: str) -> dict:
        def mae(abs_sum: float, count: int) -> float | str:
            return "" if count == 0 else abs_sum / count * 1000.0

        def rmse(sq_sum: float, count: int) -> float | str:
            return "" if count == 0 else math.sqrt(sq_sum / count) * 1000.0

        return {
            "split": split,
            "group": group,
            "frames": self.frames,
            "atoms": self.atoms,
            "energy_mae_mev_atom": mae(self.energy_abs_sum, self.energy_count),
            "energy_rmse_mev_atom": rmse(self.energy_sq_sum, self.energy_count),
            "force_component_mae_mev_A": mae(self.force_abs_sum, self.force_component_count),
            "force_component_rmse_mev_A": rmse(self.force_sq_sum, self.force_component_count),
            "force_vector_mae_mev_A": mae(self.force_vector_abs_sum, self.force_vector_count),
            "force_vector_rmse_mev_A": rmse(self.force_vector_sq_sum, self.force_vector_count),
            "missing_energy_frames": self.missing_energy,
            "missing_force_frames": self.missing_forces,
        }


def vector_triplets(atom_lines: list[str], props: dict[str, tuple[int, int]], key: str) -> list[tuple[float, float, float]]:
    start, count = props[key]
    if count != 3:
        raise ValueError(f"{key} has count {count}, expected 3")
    values = []
    for line in atom_lines:
        parts = line.split()
        values.append((float(parts[start]), float(parts[start + 1]), float(parts[start + 2])))
    return values


def score_file(split: str, path: Path, ref_energy_key: str, pred_energy_key: str, ref_forces_key: str, pred_forces_key: str) -> list[dict]:
    accs = defaultdict(Accumulator)
    first_keys_reported = False
    used_pred_energy_key = None
    used_pred_forces_key = None

    for frame_index, natoms, comment, atom_lines in iter_frames(path):
        info = parse_info(comment)
        try:
            props = parse_properties(comment)
        except Exception as exc:
            preview = comment[:240].replace("\t", " ")
            raise ValueError(
                f"{path}: frame {frame_index} has invalid extxyz comment line: {exc}; "
                f"comment preview={preview!r}"
            ) from exc
        config_type = info.get("config_type", "Default")
        groups = ["overall", f"config_type:{config_type}"]
        frame_accs = [accs[group] for group in groups]
        for acc in frame_accs:
            acc.frames += 1
            acc.atoms += natoms

        energy_key = choose_energy_key(info, pred_energy_key, ref_energy_key)
        forces_key = choose_force_key(props, pred_forces_key, ref_forces_key)
        if not first_keys_reported:
            used_pred_energy_key = energy_key
            used_pred_forces_key = forces_key
            first_keys_reported = True

        ref_energy = float_info(info, ref_energy_key)
        pred_energy = float_info(info, energy_key) if energy_key else None
        energy_diff_per_atom = None
        if ref_energy is not None and pred_energy is not None:
            energy_diff_per_atom = (pred_energy - ref_energy) / natoms

        force_diffs = None
        if ref_forces_key in props and forces_key:
            ref_forces = vector_triplets(atom_lines, props, ref_forces_key)
            pred_forces = vector_triplets(atom_lines, props, forces_key)
            force_diffs = [
                (p[0] - r[0], p[1] - r[1], p[2] - r[2])
                for r, p in zip(ref_forces, pred_forces)
            ]

        for acc in frame_accs:
            acc.add_energy(energy_diff_per_atom)
            acc.add_force_components(force_diffs)

    rows = []
    for group in sorted(accs, key=lambda item: (item != "overall", item)):
        row = accs[group].row(split, group)
        row["pred_energy_key"] = used_pred_energy_key or ""
        row["pred_forces_key"] = used_pred_forces_key or ""
        row["file"] = str(path)
        rows.append(row)
    return rows


def parse_pred_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use SPLIT=PATH, for example train=train_pred.extxyz")
    split, path = value.split("=", 1)
    split = split.strip()
    if not split:
        raise argparse.ArgumentTypeError("split name cannot be empty")
    return split, Path(path)


def fmt(value) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "split",
        "group",
        "frames",
        "atoms",
        "energy_mae_mev_atom",
        "energy_rmse_mev_atom",
        "force_component_mae_mev_A",
        "force_component_rmse_mev_A",
        "force_vector_mae_mev_A",
        "force_vector_rmse_mev_A",
        "missing_energy_frames",
        "missing_force_frames",
        "pred_energy_key",
        "pred_forces_key",
        "file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict]) -> None:
    overall = [row for row in rows if row["group"] == "overall"]
    lines = ["# MACE Split Error Summary", ""]
    lines.append("## Overall")
    lines.append("")
    lines.append("| split | frames | E MAE meV/atom | E RMSE meV/atom | F comp MAE meV/A | F comp RMSE meV/A | F vector RMSE meV/A |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in overall:
        lines.append(
            f"| {row['split']} | {row['frames']} | {fmt(row['energy_mae_mev_atom'])} | {fmt(row['energy_rmse_mev_atom'])} | "
            f"{fmt(row['force_component_mae_mev_A'])} | {fmt(row['force_component_rmse_mev_A'])} | {fmt(row['force_vector_rmse_mev_A'])} |"
        )

    lines.append("")
    lines.append("## By Config Type")
    lines.append("")
    lines.append("| split | group | frames | E RMSE meV/atom | F comp RMSE meV/A | missing E | missing F |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        if row["group"] == "overall":
            continue
        lines.append(
            f"| {row['split']} | {row['group']} | {row['frames']} | {fmt(row['energy_rmse_mev_atom'])} | "
            f"{fmt(row['force_component_rmse_mev_A'])} | {row['missing_energy_frames']} | {row['missing_force_frames']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score MACE predicted extxyz files.")
    parser.add_argument("--pred", action="append", type=parse_pred_arg, required=True, help="SPLIT=pred.extxyz. Repeat for train/valid/test.")
    parser.add_argument("--ref-energy-key", default="REF_energy")
    parser.add_argument("--pred-energy-key", default="MACE_energy")
    parser.add_argument("--ref-forces-key", default="REF_forces")
    parser.add_argument("--pred-forces-key", default="MACE_forces")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for split, path in args.pred:
        resolved = path.resolve()
        if not resolved.is_file():
            raise SystemExit(f"Prediction file does not exist: {resolved}")
        rows.extend(
            score_file(
                split,
                resolved,
                args.ref_energy_key,
                args.pred_energy_key,
                args.ref_forces_key,
                args.pred_forces_key,
            )
        )

    write_csv(args.out_csv.resolve(), rows)
    if args.out_md:
        write_md(args.out_md.resolve(), rows)

    for row in rows:
        if row["group"] == "overall":
            print(
                f"{row['split']}: frames={row['frames']} "
                f"E_RMSE={fmt(row['energy_rmse_mev_atom'])} meV/atom "
                f"F_RMSE={fmt(row['force_component_rmse_mev_A'])} meV/A "
                f"keys=({row['pred_energy_key']}, {row['pred_forces_key']})"
            )
    print(f"CSV: {args.out_csv.resolve()}")
    if args.out_md:
        print(f"Markdown: {args.out_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
