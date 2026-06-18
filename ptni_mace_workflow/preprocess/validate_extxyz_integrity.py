#!/usr/bin/env python
"""Validate basic extxyz frame structure and Properties fields."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def validate_file(path: Path, require_properties: bool = True) -> tuple[bool, int, str]:
    frame_index = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                return True, frame_index, ""
            if not natoms_line.strip():
                continue
            try:
                natoms = int(natoms_line.strip())
            except ValueError:
                return False, frame_index, f"invalid natoms line: {natoms_line[:200]!r}"

            comment = handle.readline()
            if not comment:
                return False, frame_index, "missing comment line"
            if require_properties and "Properties=" not in comment:
                return False, frame_index, f"comment has no Properties=: {comment[:240]!r}"

            for atom_i in range(natoms):
                line = handle.readline()
                if not line:
                    return False, frame_index, f"incomplete atom block at atom {atom_i}/{natoms}"
            frame_index += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate extxyz files and locate first bad frame.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--allow-missing-properties", action="store_true")
    args = parser.parse_args()

    failed = 0
    total_frames = 0
    for path in args.files:
        ok, frames, message = validate_file(path.resolve(), not args.allow_missing_properties)
        total_frames += frames
        if ok:
            print(f"OK     {path}  frames={frames}")
        else:
            failed += 1
            print(f"FAILED {path}  first_bad_frame={frames}  {message}")
    print(f"Checked files: {len(args.files)}")
    print(f"Total complete frames before failures/end: {total_frames}")
    print(f"Failed files: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
