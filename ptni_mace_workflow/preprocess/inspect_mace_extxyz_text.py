#!/usr/bin/env python
"""Inspect a MACE extxyz dataset without ASE/NumPy.

This is a streaming text parser intended for large extxyz files. It verifies
basic frame structure and summarizes keys useful for MACE training.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


KEY_RE = re.compile(r'(\w+)=(".*?"|\S+)')


def parse_info(comment: str) -> dict[str, str]:
    info = {}
    for key, value in KEY_RE.findall(comment):
        info[key] = value.strip('"')
    return info


def composition_from_atom_lines(lines: list[str]) -> str:
    symbols = []
    seen = set()
    for line in lines:
        if not line.strip():
            continue
        symbol = line.split()[0]
        if symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)

    priority = ["Pt", "Ni", "H", "C", "N", "O", "S", "P", "F", "Cl"]
    ordered = [symbol for symbol in priority if symbol in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return "".join(ordered)


def iter_frames(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        frame_index = 0
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            natoms_line = natoms_line.strip()
            if not natoms_line:
                continue
            try:
                natoms = int(natoms_line)
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


def add_numeric(value: str | None, values: list[float]) -> None:
    if value is None:
        return
    try:
        values.append(float(value))
    except ValueError:
        return


def summarize_numeric(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {"count": 0, "min": "", "max": "", "mean": ""}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect large MACE extxyz text datasets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--sample-groups", type=int, default=20)
    args = parser.parse_args()

    path = args.input.resolve()
    if not path.is_file():
        raise SystemExit(f"Input file does not exist: {path}")

    natoms_counts = Counter()
    config_type_counts = Counter()
    composition_counts = Counter()
    group_counts = Counter()
    missing_energy = 0
    missing_forces = 0
    energies = []
    max_forces = []
    frame_count = 0

    for _, natoms, comment, atom_lines in iter_frames(path):
        info = parse_info(comment)
        frame_count += 1
        natoms_counts[natoms] += 1
        config_type_counts[info.get("config_type", "Default")] += 1
        composition_counts[composition_from_atom_lines(atom_lines)] += 1
        group_counts[info.get("source_group", info.get("source_name", "ungrouped"))] += 1
        if "REF_energy" not in info:
            missing_energy += 1
        if "REF_forces" not in comment:
            missing_forces += 1
        add_numeric(info.get("REF_energy"), energies)
        add_numeric(info.get("max_force_eVA"), max_forces)

    if args.out_csv:
        rows = []
        for name, counter in (
            ("natoms", natoms_counts),
            ("config_type", config_type_counts),
            ("composition", composition_counts),
        ):
            for key, count in counter.most_common():
                rows.append({"category": name, "value": key, "count": count})
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["category", "value", "count"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"File: {path}")
    print(f"Frames: {frame_count}")
    print(f"Source groups: {len(group_counts)}")
    print(f"Compositions: {dict(composition_counts.most_common())}")
    print(f"Config types: {dict(config_type_counts.most_common())}")
    print(f"Natoms top: {dict(natoms_counts.most_common(10))}")
    print(f"Missing REF_energy frames: {missing_energy}")
    print(f"Frames whose comment lacks REF_forces property: {missing_forces}")
    print(f"REF_energy stats: {summarize_numeric(energies)}")
    print(f"max_force_eVA stats: {summarize_numeric(max_forces)}")
    print(f"Sample source groups: {dict(group_counts.most_common(args.sample_groups))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
