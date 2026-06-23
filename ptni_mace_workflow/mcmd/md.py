#!/usr/bin/env python
"""ASE MD segments for the MCMD prototype."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .neb import max_force


def _ensure_velocities(atoms: Any, temperature_K: float) -> None:
    import numpy as np
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    momenta = atoms.get_momenta()
    if momenta is None or not np.any(momenta):
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
        Stationary(atoms)
        ZeroRotation(atoms)


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_md_segment(
    atoms: Any,
    calc: Any,
    step_index: int,
    temperature_K: float,
    steps: int,
    timestep_fs: float,
    ensemble: str,
    friction_per_fs: float,
    log_csv: Path,
    trajectory_extxyz: Path | None,
    write_interval: int,
) -> Any:
    """Run one MD segment in place and return the same Atoms object."""

    if steps <= 0:
        return atoms

    from ase import units
    from ase.io import write
    from ase.md.langevin import Langevin
    from ase.md.verlet import VelocityVerlet

    atoms.calc = calc
    _ensure_velocities(atoms, temperature_K)

    if ensemble == "nve":
        dyn = VelocityVerlet(atoms, timestep_fs * units.fs)
    elif ensemble == "langevin":
        dyn = Langevin(
            atoms,
            timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=friction_per_fs / units.fs,
        )
    else:
        raise ValueError(f"unknown MD ensemble: {ensemble}")

    interval = max(int(write_interval), 1)

    def record() -> None:
        md_step = int(getattr(dyn, "nsteps", 0))
        energy = float(atoms.get_potential_energy())
        kinetic = float(atoms.get_kinetic_energy())
        row = {
            "mcmd_step": step_index,
            "md_step": md_step,
            "potential_energy_eV": f"{energy:.12f}",
            "kinetic_energy_eV": f"{kinetic:.12f}",
            "total_energy_eV": f"{energy + kinetic:.12f}",
            "temperature_K": f"{atoms.get_temperature():.6f}",
            "fmax_eVA": f"{max_force(atoms):.12f}",
        }
        _append_row(log_csv, row)
        if trajectory_extxyz is not None:
            image = atoms.copy()
            image.info.update(
                {
                    "mcmd_step": step_index,
                    "md_step": md_step,
                    "MACE_energy": energy,
                    "temperature_K": atoms.get_temperature(),
                }
            )
            write(trajectory_extxyz.as_posix(), image, format="extxyz", append=trajectory_extxyz.exists())

    record()
    dyn.attach(record, interval=interval)
    dyn.run(steps)
    record()
    return atoms

