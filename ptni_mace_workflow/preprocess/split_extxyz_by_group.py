#!/usr/bin/env python
"""Split an extxyz dataset by source group to reduce train/test leakage."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path


def group_value(atoms, key: str, fallback_index: int) -> str:
    value = atoms.info.get(key) or atoms.info.get("source_path")
    if value is None:
        return f"ungrouped_{fallback_index}"
    return str(value)


def describe(name: str, configs) -> str:
    types = Counter(str(a.info.get("config_type", "Default")) for a in configs)
    return f"{name}: {len(configs)} configs, config_type={dict(types)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Group-aware train/valid/test split for extxyz.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--group-key", default="source_group")
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    if args.valid_fraction < 0 or args.test_fraction < 0:
        raise SystemExit("fractions must be non-negative")
    if args.valid_fraction + args.test_fraction >= 1:
        raise SystemExit("valid_fraction + test_fraction must be < 1")

    from ase.io import iread, write

    configs = list(iread(args.input.as_posix(), index=":"))
    if not configs:
        raise SystemExit(f"No configurations read from {args.input}")

    grouped = defaultdict(list)
    for i, atoms in enumerate(configs):
        grouped[group_value(atoms, args.group_key, i)].append(atoms)

    group_names = list(grouped)
    rng = random.Random(args.seed)
    rng.shuffle(group_names)

    total = len(configs)
    target_test = total * args.test_fraction
    target_valid = total * args.valid_fraction
    train, valid, test = [], [], []

    for group in group_names:
        bucket = grouped[group]
        if len(test) < target_test:
            test.extend(bucket)
        elif len(valid) < target_valid:
            valid.extend(bucket)
        else:
            train.extend(bucket)

    for path, subset in ((args.train, train), (args.valid, valid), (args.test, test)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if subset:
            write(path.as_posix(), subset, format="extxyz")

    print(f"Groups: {len(group_names)}")
    print(describe("train", train))
    print(describe("valid", valid))
    print(describe("test", test))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
