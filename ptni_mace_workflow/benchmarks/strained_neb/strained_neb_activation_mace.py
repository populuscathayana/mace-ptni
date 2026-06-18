#!/usr/bin/env python
"""Predict strain-dependent activation energies with MACE CI-NEB.

The script takes IS/TS/FS POSCARs from a NEB path, applies a prescribed strain,
relaxes the endpoints at fixed strained cell, builds a NEB path using the TS as
the central initial image, runs MACE CI-NEB, and reports forward/reverse barriers.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_scan(text: str) -> list[float]:
    """Parse strain list. Use percent values, e.g. -3:3:1 or -2,-1,0,1,2."""
    if ":" in text:
        parts = [float(part) for part in text.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("strain scan must be start:stop:step in percent")
        start, stop, step = parts
        if step <= 0:
            raise argparse.ArgumentTypeError("strain step must be positive")
        values = []
        current = start
        while current <= stop + step * 1e-9:
            values.append(round(current / 100.0, 10))
            current += step
        return values
    return [float(part.strip()) / 100.0 for part in text.split(",") if part.strip()]


def strain_matrix(mode: str, strain: float):
    import numpy as np

    factors = np.ones(3)
    if mode == "isotropic":
        factors[:] = 1.0 + strain
    elif mode == "xy":
        factors[0] = 1.0 + strain
        factors[1] = 1.0 + strain
    elif mode == "x":
        factors[0] = 1.0 + strain
    elif mode == "y":
        factors[1] = 1.0 + strain
    elif mode == "z":
        factors[2] = 1.0 + strain
    else:
        raise ValueError(f"unknown strain mode: {mode}")
    return np.diag(factors)


def apply_strain(atoms, strain: float, mode: str):
    atoms = atoms.copy()
    cell = atoms.cell.array
    deformation = strain_matrix(mode, strain)
    new_cell = deformation @ cell
    atoms.set_cell(new_cell, scale_atoms=True)
    atoms.wrap()
    return atoms


def mic_scaled_delta(delta, pbc):
    import numpy as np

    delta = np.asarray(delta, dtype=float).copy()
    for axis, periodic in enumerate(pbc):
        if periodic:
            delta[:, axis] -= np.round(delta[:, axis])
    return delta


def interpolate_atoms(a0, a1, t: float):
    atoms = a0.copy()
    s0 = a0.get_scaled_positions()
    s1 = a1.get_scaled_positions()
    delta = mic_scaled_delta(s1 - s0, a0.pbc)
    atoms.set_scaled_positions(s0 + t * delta)
    atoms.wrap()
    return atoms


def build_images(is_atoms, ts_atoms, fs_atoms, n_images: int):
    if n_images < 3:
        raise ValueError("--n-images must be >= 3")
    if n_images % 2 == 0:
        raise ValueError("--n-images must be odd so the provided TS is the central image")

    center = n_images // 2
    images = []
    for i in range(n_images):
        if i == 0:
            images.append(is_atoms.copy())
        elif i == n_images - 1:
            images.append(fs_atoms.copy())
        elif i == center:
            images.append(ts_atoms.copy())
        elif i < center:
            images.append(interpolate_atoms(is_atoms, ts_atoms, i / center))
        else:
            images.append(interpolate_atoms(ts_atoms, fs_atoms, (i - center) / center))
    return images


def attach_calc(atoms_or_images, calc):
    if isinstance(atoms_or_images, list):
        for atoms in atoms_or_images:
            atoms.calc = calc
    else:
        atoms_or_images.calc = calc


def max_force(atoms) -> float:
    import numpy as np

    forces = atoms.get_forces()
    return float(np.linalg.norm(forces, axis=1).max())


def relax_atoms(atoms, calc, fmax: float, steps: int, trajectory: Path | None = None):
    from ase.optimize import FIRE, BFGS

    attach_calc(atoms, calc)
    opt_cls = FIRE if steps > 0 else BFGS
    opt = opt_cls(atoms, trajectory=str(trajectory) if trajectory else None, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    energy = float(atoms.get_potential_energy())
    force = max_force(atoms)
    atoms.calc = None
    return atoms, energy, force


def run_neb(images, calc, fmax: float, steps: int, climb: bool, trajectory: Path | None = None):
    from ase.mep import NEB
    from ase.optimize import FIRE

    attach_calc(images, calc)
    neb = NEB(images, climb=climb, allow_shared_calculator=True)
    opt = FIRE(neb, trajectory=str(trajectory) if trajectory else None, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    energies = [float(image.get_potential_energy()) for image in images]
    forces = [max_force(image) for image in images]
    for image in images:
        image.calc = None
    return energies, forces


def write_images(path: Path, images):
    from ase.io import write

    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), images, format="extxyz")


def main() -> int:
    parser = argparse.ArgumentParser(description="MACE strain-dependent activation energy via CI-NEB.")
    parser.add_argument("--is-poscar", type=Path, required=True)
    parser.add_argument("--ts-poscar", type=Path, required=True)
    parser.add_argument("--fs-poscar", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="MACE .model file, not compiled.model.")
    parser.add_argument("--strain", type=parse_scan, default=parse_scan("-3:3:1"), help="Percent strain scan, e.g. -3:3:1.")
    parser.add_argument(
        "--mode",
        choices=["isotropic", "xy", "x", "y", "z"],
        default="isotropic",
        help="Strain mode. For bulk use isotropic; for slab usually use xy.",
    )
    parser.add_argument("--n-images", type=int, default=5, help="Odd number including endpoints. Default: 5.")
    parser.add_argument("--endpoint-fmax", type=float, default=0.02)
    parser.add_argument("--endpoint-steps", type=int, default=300)
    parser.add_argument("--neb-fmax", type=float, default=0.05)
    parser.add_argument("--neb-steps", type=int, default=400)
    parser.add_argument("--no-climb", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", default="float64")
    parser.add_argument("--out-dir", type=Path, default=Path("mace_workspace/runs/benchmarks/strained_neb/manual"))
    parser.add_argument("--name", default="pt_vacancy_strained_neb")
    parser.add_argument("--write-images", action="store_true")
    args = parser.parse_args()

    from ase.io import read
    from mace.calculators import MACECalculator

    is0 = read(str(args.is_poscar), format="vasp")
    ts0 = read(str(args.ts_poscar), format="vasp")
    fs0 = read(str(args.fs_poscar), format="vasp")

    if len({len(is0), len(ts0), len(fs0)}) != 1:
        raise SystemExit("IS/TS/FS atom counts differ")
    if is0.get_chemical_symbols() != ts0.get_chemical_symbols() or is0.get_chemical_symbols() != fs0.get_chemical_symbols():
        raise SystemExit("IS/TS/FS chemical symbol order differs")

    calc = MACECalculator(
        model_paths=str(args.model.resolve()),
        device=args.device,
        default_dtype=args.default_dtype,
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for strain in args.strain:
        label = f"{strain * 100:+.3f}pct".replace("+", "p").replace("-", "m").replace(".", "p")
        strain_dir = out_dir / f"{args.name}_{args.mode}_{label}"
        strain_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== strain {strain * 100:+.3f}% ({args.mode}) ===")

        is_s = apply_strain(is0, strain, args.mode)
        ts_s = apply_strain(ts0, strain, args.mode)
        fs_s = apply_strain(fs0, strain, args.mode)

        # Single-point barrier using the provided TS before any relaxation/NEB.
        attach_calc([is_s, ts_s, fs_s], calc)
        e_is_sp = float(is_s.get_potential_energy())
        e_ts_sp = float(ts_s.get_potential_energy())
        e_fs_sp = float(fs_s.get_potential_energy())
        f_ts_sp = max_force(ts_s)
        for atoms in (is_s, ts_s, fs_s):
            atoms.calc = None

        is_relaxed, e_is, f_is = relax_atoms(
            is_s, calc, args.endpoint_fmax, args.endpoint_steps, strain_dir / "is_relax.traj"
        )
        fs_relaxed, e_fs, f_fs = relax_atoms(
            fs_s, calc, args.endpoint_fmax, args.endpoint_steps, strain_dir / "fs_relax.traj"
        )

        images = build_images(is_relaxed, ts_s, fs_relaxed, args.n_images)
        pre_neb_path = strain_dir / "neb_initial.extxyz"
        final_neb_path = strain_dir / "neb_final.extxyz"
        if args.write_images:
            write_images(pre_neb_path, images)

        energies, forces = run_neb(
            images,
            calc,
            args.neb_fmax,
            args.neb_steps,
            climb=not args.no_climb,
            trajectory=strain_dir / "neb.traj",
        )
        if args.write_images:
            write_images(final_neb_path, images)

        emax = max(energies)
        imax = energies.index(emax)
        barrier_forward = emax - energies[0]
        barrier_reverse = emax - energies[-1]
        reaction_energy = energies[-1] - energies[0]

        row = {
            "strain_fraction": strain,
            "strain_percent": strain * 100.0,
            "mode": args.mode,
            "n_atoms": len(is0),
            "n_images": args.n_images,
            "sp_E_is_eV": e_is_sp,
            "sp_E_ts_eV": e_ts_sp,
            "sp_E_fs_eV": e_fs_sp,
            "sp_barrier_forward_eV": e_ts_sp - e_is_sp,
            "sp_barrier_reverse_eV": e_ts_sp - e_fs_sp,
            "sp_reaction_eV": e_fs_sp - e_is_sp,
            "sp_ts_max_force_eVA": f_ts_sp,
            "relaxed_E_is_eV": e_is,
            "relaxed_E_fs_eV": e_fs,
            "relaxed_is_max_force_eVA": f_is,
            "relaxed_fs_max_force_eVA": f_fs,
            "neb_E_initial_eV": energies[0],
            "neb_E_final_eV": energies[-1],
            "neb_E_max_eV": emax,
            "neb_ts_image_index": imax,
            "neb_barrier_forward_eV": barrier_forward,
            "neb_barrier_reverse_eV": barrier_reverse,
            "neb_reaction_eV": reaction_energy,
            "neb_max_force_path_eVA": max(forces),
            "neb_max_force_ts_image_eVA": forces[imax],
            "neb_energies_relative_eV": ";".join(f"{e - energies[0]:.10f}" for e in energies),
            "neb_forces_max_eVA": ";".join(f"{f:.10f}" for f in forces),
            "strain_dir": str(strain_dir),
        }
        rows.append(row)

        print(
            f"SP barrier={row['sp_barrier_forward_eV']:.6f} eV; "
            f"NEB barrier={barrier_forward:.6f} eV; "
            f"reaction={reaction_energy:.6f} eV; "
            f"TS image={imax}; max path force={row['neb_max_force_path_eVA']:.4f} eV/A"
        )

    csv_path = out_dir / f"{args.name}_{args.mode}_summary.csv"
    md_path = out_dir / f"{args.name}_{args.mode}_summary.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Strain-Dependent MACE NEB Activation Test",
        "",
        f"- Model: `{args.model.resolve()}`",
        f"- IS: `{args.is_poscar.resolve()}`",
        f"- TS: `{args.ts_poscar.resolve()}`",
        f"- FS: `{args.fs_poscar.resolve()}`",
        f"- Mode: `{args.mode}`",
        f"- Images: {args.n_images}",
        f"- Endpoint relax: fmax={args.endpoint_fmax}, steps={args.endpoint_steps}",
        f"- NEB: fmax={args.neb_fmax}, steps={args.neb_steps}, climb={not args.no_climb}",
        "",
        "| strain % | SP barrier eV | NEB barrier eV | reaction eV | TS image | max path force eV/A |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strain_percent']:.3f} | {row['sp_barrier_forward_eV']:.6f} | "
            f"{row['neb_barrier_forward_eV']:.6f} | {row['neb_reaction_eV']:.6f} | "
            f"{row['neb_ts_image_index']} | {row['neb_max_force_path_eVA']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `SP barrier` uses the supplied TS POSCAR directly after applying strain; it is a cheap diagnostic.",
            "- `NEB barrier` relaxes endpoints and then runs CI-NEB using the supplied TS as the central initial image.",
            "- For TS validation, compare against DFT NEB/CI-NEB at selected strains; MACE-only NEB checks internal PES consistency but is not a DFT benchmark by itself.",
            "- If endpoint and TS DFT convergence criteria differ, analyze endpoint energy errors and TS/path errors separately.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nCSV: {csv_path}")
    print(f"Summary: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
