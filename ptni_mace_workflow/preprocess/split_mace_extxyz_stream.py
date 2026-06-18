#!/usr/bin/env python
"""Stream-split extxyz into train/valid/test by source_group.

This avoids loading the full dataset and does not require ASE. The assignment is
deterministic: all frames with the same group key go to the same split.
"""

from __future__ import annotations

import argparse
import hashlib
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
            yield frame_index, natoms_line, comment, atom_lines
            frame_index += 1


def split_for_group(group: str, valid_percent: int, test_percent: int, seed: str) -> str:
    digest = hashlib.sha1(f"{seed}|{group}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + valid_percent:
        return "valid"
    return "train"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream split MACE extxyz by source_group.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--group-key", default="source_group")
    parser.add_argument("--valid-percent", type=int, default=10)
    parser.add_argument("--test-percent", type=int, default=10)
    parser.add_argument("--seed", default="20260608")
    args = parser.parse_args()

    if args.valid_percent < 0 or args.test_percent < 0:
        raise SystemExit("percent values must be non-negative")
    if args.valid_percent + args.test_percent >= 100:
        raise SystemExit("valid-percent + test-percent must be < 100")

    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "train.extxyz",
        "valid": out_dir / "valid.extxyz",
        "test": out_dir / "test.extxyz",
    }

    frame_counts = Counter()
    group_to_split = {}
    split_groups = Counter()

    handles = {name: path.open("w", encoding="utf-8", newline="") for name, path in paths.items()}
    try:
        for frame_index, natoms_line, comment, atom_lines in iter_frame_text(input_path):
            info = parse_info(comment)
            group = info.get(args.group_key) or info.get("source_name") or f"frame_{frame_index}"
            split = group_to_split.get(group)
            if split is None:
                split = split_for_group(group, args.valid_percent, args.test_percent, args.seed)
                group_to_split[group] = split
                split_groups[split] += 1
            handle = handles[split]
            handle.write(natoms_line)
            handle.write(comment)
            handle.writelines(atom_lines)
            frame_counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()

    print(f"Input: {input_path}")
    print(f"Output dir: {out_dir}")
    print(f"Groups: {len(group_to_split)}")
    print(f"Frames by split: {dict(frame_counts)}")
    print(f"Groups by split: {dict(split_groups)}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
