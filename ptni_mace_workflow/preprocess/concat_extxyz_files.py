#!/usr/bin/env python
"""Concatenate extxyz files in sorted order.

This is mainly used to merge chunked prediction files back into one
train_pred.extxyz before scoring.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Concatenate extxyz files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sort", action="store_true", help="Sort input paths lexicographically before concatenating.")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    inputs = [path.resolve() for path in args.inputs]
    if args.sort:
        inputs = sorted(inputs)

    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit("Missing input files:\n" + "\n".join(str(path) for path in missing[:20]))

    empty = [path for path in inputs if path.stat().st_size == 0]
    if empty and not args.allow_empty:
        raise SystemExit("Empty input files:\n" + "\n".join(str(path) for path in empty[:20]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    with args.output.resolve().open("wb") as out:
        for path in inputs:
            size = path.stat().st_size
            if size == 0:
                continue
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, out, length=32 * 1024 * 1024)
            total_bytes += size

    print(f"Inputs: {len(inputs)}")
    print(f"Output: {args.output.resolve()}")
    print(f"Bytes written: {total_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
