#!/usr/bin/env python
"""Filter a MACE extxyz dataset using metadata in the comment line.

The main use here is dropping very high-force ionic steps before a first MACE
fine-tuning attempt. The parser is streaming and does not require ASE.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


KEY_RE = re.compile(r'(\w+)=(".*?"|\S+)')


def parse_info(comment: str) -> dict[str, str]:
    return {key: value.strip('"') for key, value in KEY_RE.findall(comment)}


def iter_frame_text(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        frame_index = 0
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            if not natoms_line.strip():
                continue
            natoms = int(natoms_line.strip())
            comment = handle.readline()
            if not comment:
                raise ValueError(f"frame {frame_index}: missing comment line")
            atom_lines = [handle.readline() for _ in range(natoms)]
            if len(atom_lines) != natoms or any(line == "" for line in atom_lines):
                raise ValueError(f"frame {frame_index}: incomplete atom block")
            yield frame_index, natoms_line, comment, atom_lines
            frame_index += 1


def get_float(info: dict[str, str], key: str) -> float | None:
    if key not in info:
        return None
    try:
        return float(info[key])
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter MACE extxyz by max_force_eVA.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-force", type=float, default=20.0)
    parser.add_argument(
        "--missing-max-force",
        choices=["keep", "drop"],
        default="keep",
        help="What to do if max_force_eVA is absent. Default: keep.",
    )
    parser.add_argument("--report-csv", type=Path, default=None)
    args = parser.parse_args()

    counts = Counter()
    by_config = Counter()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="") as out:
        for frame_index, natoms_line, comment, atom_lines in iter_frame_text(args.input.resolve()):
            info = parse_info(comment)
            config_type = info.get("config_type", "Default")
            max_force = get_float(info, "max_force_eVA")
            keep = True
            reason = "kept"
            if max_force is None:
                if args.missing_max_force == "drop":
                    keep = False
                    reason = "missing_max_force"
            elif max_force > args.max_force:
                keep = False
                reason = "max_force"

            counts[reason] += 1
            by_config[f"{config_type}:{reason}"] += 1
            if keep:
                out.write(natoms_line)
                out.write(comment)
                out.writelines(atom_lines)

            if args.report_csv is not None:
                rows.append(
                    {
                        "frame_index": frame_index,
                        "config_type": config_type,
                        "source_group": info.get("source_group", ""),
                        "max_force_eVA": "" if max_force is None else max_force,
                        "status": reason,
                    }
                )

    if args.report_csv is not None:
        args.report_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.report_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["frame_index", "config_type", "source_group", "max_force_eVA", "status"],
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f"Input: {args.input.resolve()}")
    print(f"Output: {args.output.resolve()}")
    print(f"Max force threshold: {args.max_force}")
    print(f"Counts: {dict(counts)}")
    print(f"By config/status: {dict(by_config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
