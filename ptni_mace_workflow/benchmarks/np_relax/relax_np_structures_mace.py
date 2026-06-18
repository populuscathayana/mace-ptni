#!/usr/bin/env python
"""Relax NP structures with a MACE model and log every optimization step."""

from __future__ import annotations

import argparse
import csv
import inspect
import re
from pathlib import Path
from typing import Any


def safe_label(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "structure"


def max_force(atoms: Any) -> float:
    import numpy as np

    forces = atoms.get_forces()
    return float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0


def composition(atoms: Any) -> str:
    from collections import Counter

    counts = Counter(atoms.get_chemical_symbols())
    return "".join(f"{el}{counts[el]}" for el in sorted(counts))


def set_pbc(atoms: Any, mode: str) -> None:
    if mode == "true":
        atoms.pbc = (True, True, True)
    elif mode == "false":
        atoms.pbc = (False, False, False)


def wrap_scaled_positions(atoms: Any) -> None:
    import numpy as np

    scaled = atoms.get_scaled_positions(wrap=False)
    atoms.set_scaled_positions(np.mod(scaled, 1.0))


def make_optimizer(args: argparse.Namespace, atoms: Any, trajectory: Path | None, logfile: Path | None) -> Any:
    from ase.optimize import BFGS, FIRE, LBFGS

    classes = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}
    opt_cls = classes[args.optimizer.upper()]
    kwargs: dict[str, Any] = {
        "trajectory": str(trajectory) if trajectory else None,
        "logfile": str(logfile) if logfile else None,
    }
    signature = inspect.signature(opt_cls)
    if args.maxstep is not None and "maxstep" in signature.parameters:
        kwargs["maxstep"] = args.maxstep
    if args.optimizer.upper() == "FIRE" and "downhill_check" in signature.parameters:
        kwargs["downhill_check"] = args.fire_downhill_check
    return opt_cls(atoms, **kwargs)


def write_step_row(handle: Any, writer: csv.DictWriter, row: dict[str, Any]) -> None:
    writer.writerow(row)
    handle.flush()


def relax_one(path: Path, calc: Any, args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    from ase.io import read, write

    label = safe_label(path.stem)
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    step_csv = out_dir / "relax_steps.csv"
    final_vasp = out_dir / f"{label}_relaxed.vasp"
    final_extxyz = out_dir / f"{label}_relaxed.extxyz"
    traj_extxyz = out_dir / f"{label}_relax_path.extxyz"
    ase_log = out_dir / "ase_optimizer.log"
    ase_traj = out_dir / "ase_optimizer.traj"

    if step_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{step_csv} exists. Use --overwrite or choose a new --out-dir.")
    if traj_extxyz.exists() and args.write_trajectory:
        traj_extxyz.unlink()

    atoms = read(path.as_posix(), format=args.input_format)
    if args.wrap_scaled:
        wrap_scaled_positions(atoms)
    set_pbc(atoms, args.pbc)
    atoms.calc = calc

    fields = [
        "step",
        "energy_eV",
        "energy_per_atom_eV",
        "fmax_eVA",
        "natoms",
        "composition",
        "wrap_scaled",
        "pbc_mode",
    ]

    step_state = {"step": 0}
    comp = composition(atoms)

    with step_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def record_step(step: int) -> None:
            step_state["step"] = step
            energy = float(atoms.get_potential_energy())
            fmax = max_force(atoms)
            row = {
                "step": step,
                "energy_eV": f"{energy:.12f}",
                "energy_per_atom_eV": f"{energy / len(atoms):.12f}",
                "fmax_eVA": f"{fmax:.12f}",
                "natoms": len(atoms),
                "composition": comp,
                "wrap_scaled": args.wrap_scaled,
                "pbc_mode": str(tuple(bool(x) for x in atoms.pbc)),
            }
            write_step_row(handle, writer, row)
            print(
                f"[{label}] step={step:04d} "
                f"E={energy:.12f} eV "
                f"E/atom={energy / len(atoms):.12f} eV "
                f"fmax={fmax:.6f} eV/A",
                flush=True,
            )
            if args.write_trajectory:
                image = atoms.copy()
                image.info["step"] = step
                image.info["MACE_energy"] = energy
                image.info["MACE_fmax_eVA"] = fmax
                write(traj_extxyz.as_posix(), image, format="extxyz", append=traj_extxyz.exists())

        record_step(0)

        converged = False
        if args.steps > 0:
            opt = make_optimizer(
                args,
                atoms,
                ase_traj if args.write_ase_trajectory else None,
                ase_log if args.write_ase_log else None,
            )

            last_logged = {"step": 0}

            def after_step() -> None:
                current_step = int(getattr(opt, "nsteps", last_logged["step"]))
                if current_step <= last_logged["step"]:
                    return
                last_logged["step"] = current_step
                record_step(current_step)

            opt.attach(after_step, interval=1)
            converged = bool(opt.run(fmax=args.fmax, steps=args.steps))
        else:
            converged = max_force(atoms) <= args.fmax

    final_energy = float(atoms.get_potential_energy())
    final_fmax = max_force(atoms)
    atoms.info["MACE_energy"] = final_energy
    atoms.info["MACE_fmax_eVA"] = final_fmax
    atoms.info["optimizer_converged"] = converged
    write(final_vasp.as_posix(), atoms, format="vasp", direct=True, vasp5=True)
    write(final_extxyz.as_posix(), atoms, format="extxyz")
    atoms.calc = None

    return {
        "input": str(path),
        "label": label,
        "status": "ok",
        "natoms": len(atoms),
        "composition": comp,
        "wrap_scaled": args.wrap_scaled,
        "pbc_mode": str(tuple(bool(x) for x in atoms.pbc)),
        "converged": converged,
        "steps_recorded": step_state["step"],
        "final_energy_eV": f"{final_energy:.12f}",
        "final_energy_per_atom_eV": f"{final_energy / len(atoms):.12f}",
        "final_fmax_eVA": f"{final_fmax:.12f}",
        "step_log": str(step_csv),
        "final_vasp": str(final_vasp),
        "final_extxyz": str(final_extxyz),
        "trajectory_extxyz": str(traj_extxyz) if args.write_trajectory else "",
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_md(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# NP MACE Structure Relaxation",
        "",
        f"- Input dir: `{args.input_dir.resolve()}`",
        f"- Pattern: `{args.pattern}`",
        f"- Model: `{args.model.resolve()}`",
        f"- Device: `{args.device}`",
        f"- Optimizer: `{args.optimizer}`",
        f"- fmax: {args.fmax} eV/A",
        f"- max steps: {args.steps}",
        f"- maxstep: {args.maxstep}",
        f"- Wrap scaled coordinates before relaxation: {args.wrap_scaled}",
        f"- PBC mode: `{args.pbc}`",
        "",
        "| label | natoms | composition | converged | final E/atom eV | final fmax eV/A |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('label', '')} | {row.get('natoms', '')} | {row.get('composition', '')} | "
            f"{row.get('converged', '')} | {row.get('final_energy_per_atom_eV', '')} | "
            f"{row.get('final_fmax_eVA', '')} |"
        )
    lines.append("")
    lines.append("Each structure directory contains `relax_steps.csv`, final VASP/extxyz files, and optionally an extxyz path trajectory.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("mace_workspace/inputs/np_structures"))
    parser.add_argument("--pattern", default="*.vasp")
    parser.add_argument("--model", type=Path, default=Path("mace_workspace/models/ft_best_loss/model.model"))
    parser.add_argument("--out-dir", type=Path, default=Path("mace_workspace/runs/benchmarks/np_relax/ft_best_loss"))
    parser.add_argument("--input-format", default="vasp")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--optimizer", choices=["FIRE", "BFGS", "LBFGS"], default="FIRE")
    parser.add_argument("--fmax", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--maxstep", type=float, default=0.05)
    parser.add_argument("--fire-downhill-check", action="store_true")
    parser.add_argument("--pbc", choices=["from-input", "true", "false"], default="false")
    parser.add_argument("--no-wrap-scaled", dest="wrap_scaled", action="store_false", help="Do not first map scaled coordinates into [0,1).")
    parser.set_defaults(wrap_scaled=True)
    parser.add_argument("--max-structures", type=int, default=None)
    parser.add_argument("--write-trajectory", action="store_true")
    parser.add_argument("--write-ase-trajectory", action="store_true")
    parser.add_argument("--write-ase-log", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    paths = sorted(path for path in args.input_dir.glob(args.pattern) if path.is_file())
    if args.max_structures is not None:
        paths = paths[: args.max_structures]
    if not paths:
        raise SystemExit(f"No input files found in {args.input_dir} matching {args.pattern!r}")

    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=str(args.model),
        device=args.device,
        default_dtype=args.default_dtype,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in paths:
        print(f"\n=== Relaxing {path} ===", flush=True)
        try:
            row = relax_one(path, calc, args, args.out_dir)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            row = {
                "input": str(path),
                "label": safe_label(path.stem),
                "status": "failed",
                "error": repr(exc),
            }
            print(f"[{row['label']}] FAILED: {exc}", flush=True)
        rows.append(row)

    write_summary(args.out_dir / "np_relax_summary.csv", rows)
    write_md(args.out_dir / "np_relax_summary.md", rows, args)
    print(f"\nSummary CSV: {(args.out_dir / 'np_relax_summary.csv').resolve()}")
    print(f"Summary MD: {(args.out_dir / 'np_relax_summary.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
