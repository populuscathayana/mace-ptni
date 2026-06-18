#!/usr/bin/env python
"""Evaluate a MACE .model on an extxyz file one configuration at a time.

This is a lower-memory fallback for cases where mace_eval_configs runs out of
GPU memory during batched/final evaluation. It writes predictions as
MACE_energy and MACE_forces so score_mace_predictions_extxyz.py can read them.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path


def ascii_safe(value):
    if isinstance(value, str):
        return value.encode("ascii", errors="replace").decode("ascii")
    return value


def sanitize_info_for_extxyz(atoms):
    """Avoid ASE extxyz ASCII writer failures from non-ASCII metadata."""
    cleaned = {}
    for key, value in atoms.info.items():
        safe_key = str(key).encode("ascii", errors="ignore").decode("ascii")
        if not safe_key:
            continue
        cleaned[safe_key] = ascii_safe(value)
    atoms.info.clear()
    atoms.info.update(cleaned)


def main() -> int:
    parser = argparse.ArgumentParser(description="Low-memory one-by-one MACE extxyz evaluation.")
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", default="float64")
    parser.add_argument("--energy-key", default="MACE_energy")
    parser.add_argument("--forces-key", default="MACE_forces")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output instead of replacing it. Default is to remove an existing output first.",
    )
    parser.add_argument(
        "--no-sanitize-info",
        action="store_true",
        help="Do not replace non-ASCII metadata before writing extxyz.",
    )
    args = parser.parse_args()

    from ase.io import iread, write
    from mace.calculators import MACECalculator

    try:
        import torch
    except Exception:
        torch = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.append:
        args.output.unlink()
    calc = MACECalculator(
        model_paths=str(args.model.resolve()),
        device=args.device,
        default_dtype=args.default_dtype,
    )

    written = 0
    first = True
    for index, atoms in enumerate(iread(str(args.configs.resolve()), index=":")):
        if args.limit is not None and index >= args.limit:
            break

        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        atoms.info[args.energy_key] = energy
        atoms.arrays[args.forces_key] = forces
        atoms.calc = None
        if not args.no_sanitize_info:
            sanitize_info_for_extxyz(atoms)

        write(str(args.output.resolve()), atoms, format="extxyz", append=not first)
        first = False
        written += 1

        if written % 25 == 0:
            print(f"Evaluated {written} configs")
        if torch is not None and args.device.startswith("cuda") and written % args.empty_cache_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Input: {args.configs.resolve()}")
    print(f"Model: {args.model.resolve()}")
    print(f"Output: {args.output.resolve()}")
    print(f"Written configs: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
