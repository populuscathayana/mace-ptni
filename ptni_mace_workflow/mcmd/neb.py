#!/usr/bin/env python
"""ASE CI-NEB helpers for vacancy-mediated hop events."""

from __future__ import annotations

import csv
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .events import HopEvent


@dataclass
class NebResult:
    event: HopEvent
    status: str
    converged: bool
    cache_hit: bool
    barrier_eV: float
    reverse_barrier_eV: float
    reaction_energy_eV: float
    energies_eV: list[float]
    max_forces_eVA: list[float]
    neb_dir: Path
    summary_path: Path
    final_atoms: Any
    message: str = ""

    def rate(self, temperature_K: float, attempt_frequency_s: float) -> float:
        if self.status != "ok" or not math.isfinite(self.barrier_eV):
            return 0.0
        k_b = 8.617333262145e-5
        effective_barrier = max(float(self.barrier_eV), 0.0)
        if temperature_K <= 0:
            return 0.0
        return float(attempt_frequency_s * math.exp(-effective_barrier / (k_b * temperature_K)))

    def to_row(self) -> dict[str, Any]:
        return {
            "neb_status": self.status,
            "neb_converged": self.converged,
            "cache_hit": self.cache_hit,
            "barrier_eV": self.barrier_eV,
            "reverse_barrier_eV": self.reverse_barrier_eV,
            "reaction_energy_eV": self.reaction_energy_eV,
            "max_force_eVA": max(self.max_forces_eVA) if self.max_forces_eVA else "",
            "neb_dir": str(self.neb_dir),
            "summary_path": str(self.summary_path),
            "message": self.message,
        }


def attach_calc(atoms_or_images: Any, calc: Any) -> None:
    if isinstance(atoms_or_images, list):
        for atoms in atoms_or_images:
            atoms.calc = calc
    else:
        atoms_or_images.calc = calc


def detach_calc(atoms_or_images: Any) -> None:
    if isinstance(atoms_or_images, list):
        for atoms in atoms_or_images:
            atoms.calc = None
    else:
        atoms_or_images.calc = None


def max_force(atoms: Any) -> float:
    import numpy as np

    forces = atoms.get_forces()
    return float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0


def interpolate_atoms(a0: Any, a1: Any, t: float) -> Any:
    atoms = a0.copy()
    positions = (1.0 - t) * a0.get_positions() + t * a1.get_positions()
    atoms.set_positions(positions)
    atoms.pbc = a0.pbc
    return atoms


def build_images(is_atoms: Any, fs_atoms: Any, n_images: int) -> list[Any]:
    if n_images < 3:
        raise ValueError("--neb-images must be >= 3")
    return [interpolate_atoms(is_atoms, fs_atoms, i / (n_images - 1)) for i in range(n_images)]


def _fire_kwargs(atoms_or_neb: Any, trajectory: Path | None, logfile: Path | None, maxstep: float | None, downhill_check: bool) -> dict[str, Any]:
    from ase.optimize import FIRE

    signature = inspect.signature(FIRE)
    kwargs: dict[str, Any] = {
        "trajectory": str(trajectory) if trajectory else None,
        "logfile": str(logfile) if logfile else None,
    }
    if maxstep is not None and "maxstep" in signature.parameters:
        kwargs["maxstep"] = maxstep
    if "downhill_check" in signature.parameters:
        kwargs["downhill_check"] = downhill_check
    return kwargs


def relax_endpoint(atoms: Any, calc: Any, fmax: float, steps: int, maxstep: float | None, downhill_check: bool) -> tuple[Any, bool]:
    from ase.optimize import FIRE

    relaxed = atoms.copy()
    attach_calc(relaxed, calc)
    try:
        opt = FIRE(relaxed, **_fire_kwargs(relaxed, None, None, maxstep, downhill_check))
        converged = bool(opt.run(fmax=fmax, steps=steps))
        return relaxed, converged
    finally:
        detach_calc(relaxed)


def _write_energy_profile(path: Path, energies: list[float], forces: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "energy_eV", "relative_energy_eV", "fmax_eVA"])
        writer.writeheader()
        e0 = energies[0] if energies else 0.0
        for index, (energy, force) in enumerate(zip(energies, forces)):
            writer.writerow(
                {
                    "image": index,
                    "energy_eV": f"{energy:.12f}",
                    "relative_energy_eV": f"{energy - e0:.12f}",
                    "fmax_eVA": f"{force:.12f}",
                }
            )


def _summary_payload(
    event: HopEvent,
    status: str,
    converged: bool,
    energies: list[float],
    forces: list[float],
    neb_dir: Path,
    message: str,
    endpoint_relax_mode: str,
) -> dict[str, Any]:
    if energies:
        barrier = float(max(energies) - energies[0])
        reverse = float(max(energies) - energies[-1])
        reaction = float(energies[-1] - energies[0])
    else:
        barrier = float("nan")
        reverse = float("nan")
        reaction = float("nan")
    return {
        "event_id": event.event_id,
        "status": status,
        "converged": converged,
        "barrier_eV": barrier,
        "reverse_barrier_eV": reverse,
        "reaction_energy_eV": reaction,
        "energies_eV": energies,
        "max_forces_eVA": forces,
        "max_force_eVA": max(forces) if forces else None,
        "endpoint_relax_mode": endpoint_relax_mode,
        "neb_dir": str(neb_dir),
        "neb_initial_extxyz": str(neb_dir / "neb_initial.extxyz"),
        "neb_final_extxyz": str(neb_dir / "neb_final.extxyz"),
        "energy_profile_csv": str(neb_dir / "energy_profile.csv"),
        "message": message,
        **event.to_row(),
    }


def neb_result_from_summary(event: HopEvent, summary: dict[str, Any], cache_root: Path) -> NebResult:
    from ase.io import read

    final_atoms = event.final_atoms
    final_path = summary.get("neb_final_extxyz")
    if final_path and Path(final_path).is_file():
        try:
            final_atoms = read(final_path)
        except Exception:
            final_atoms = event.final_atoms

    neb_dir = Path(summary.get("neb_dir", cache_root / event.event_id))
    summary_path = neb_dir / "summary.json"
    return NebResult(
        event=event,
        status=str(summary.get("status", "ok")),
        converged=bool(summary.get("converged", False)),
        cache_hit=True,
        barrier_eV=float(summary.get("barrier_eV", float("nan"))),
        reverse_barrier_eV=float(summary.get("reverse_barrier_eV", float("nan"))),
        reaction_energy_eV=float(summary.get("reaction_energy_eV", float("nan"))),
        energies_eV=[float(x) for x in summary.get("energies_eV", [])],
        max_forces_eVA=[float(x) for x in summary.get("max_forces_eVA", [])],
        neb_dir=neb_dir,
        summary_path=summary_path,
        final_atoms=final_atoms,
        message=str(summary.get("message", "")),
    )


def run_event_neb(event: HopEvent, calc: Any, args: Any, neb_dir: Path) -> NebResult:
    from ase.io import write
    from ase.mep import NEB
    from ase.optimize import FIRE
    from ptni_mace_workflow.common.paths import write_json

    neb_dir.mkdir(parents=True, exist_ok=True)
    is_atoms = event.initial_atoms.copy()
    fs_atoms = event.final_atoms.copy()

    endpoint_message = ""
    if args.endpoint_relax_mode == "full":
        is_atoms, is_conv = relax_endpoint(
            is_atoms,
            calc,
            args.endpoint_fmax,
            args.endpoint_steps,
            args.endpoint_maxstep,
            args.fire_downhill_check,
        )
        fs_atoms, fs_conv = relax_endpoint(
            fs_atoms,
            calc,
            args.endpoint_fmax,
            args.endpoint_steps,
            args.endpoint_maxstep,
            args.fire_downhill_check,
        )
        endpoint_message = f"endpoint_relax_converged={is_conv}/{fs_conv}"

    write((neb_dir / "neb_initial.extxyz").as_posix(), [is_atoms, fs_atoms], format="extxyz")
    images = build_images(is_atoms, fs_atoms, args.neb_images)
    write((neb_dir / "neb_initial_path.extxyz").as_posix(), images, format="extxyz")

    attach_calc(images, calc)
    status = "ok"
    message = endpoint_message
    try:
        neb = NEB(images, climb=args.climb, allow_shared_calculator=True, method="improvedtangent")
        opt = FIRE(
            neb,
            **_fire_kwargs(
                neb,
                neb_dir / "ase_neb.traj" if args.write_ase_trajectory else None,
                neb_dir / "ase_neb.log" if args.write_ase_log else None,
                args.neb_maxstep,
                args.fire_downhill_check,
            ),
        )
        converged = bool(opt.run(fmax=args.neb_fmax, steps=args.neb_steps))
        energies = [float(image.get_potential_energy()) for image in images]
        forces = [max_force(image) for image in images]
    except Exception as exc:
        status = "failed"
        converged = False
        energies = []
        forces = []
        message = f"{endpoint_message}; {type(exc).__name__}: {exc}".strip("; ")
    finally:
        detach_calc(images)

    write((neb_dir / "neb_final.extxyz").as_posix(), images, format="extxyz")
    _write_energy_profile(neb_dir / "energy_profile.csv", energies, forces)
    summary = _summary_payload(event, status, converged, energies, forces, neb_dir, message, args.endpoint_relax_mode)
    summary_path = neb_dir / "summary.json"
    write_json(summary_path, summary)

    return NebResult(
        event=event,
        status=status,
        converged=converged,
        cache_hit=False,
        barrier_eV=float(summary["barrier_eV"]),
        reverse_barrier_eV=float(summary["reverse_barrier_eV"]),
        reaction_energy_eV=float(summary["reaction_energy_eV"]),
        energies_eV=energies,
        max_forces_eVA=forces,
        neb_dir=neb_dir,
        summary_path=summary_path,
        final_atoms=images[-1].copy() if images else event.final_atoms,
        message=message,
    )

