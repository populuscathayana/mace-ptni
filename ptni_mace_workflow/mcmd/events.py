#!/usr/bin/env python
"""Vacancy-mediated hop event generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sites import VacancySite


@dataclass
class HopEvent:
    event_id: str
    atom_index: int
    atom_symbol: str
    vacancy: VacancySite
    atom_initial_cartesian: np.ndarray
    atom_final_cartesian: np.ndarray
    new_vacancy_cartesian: np.ndarray
    hop_distance_A: float
    direct_distance_A: float
    mic_distance_A: float
    crosses_pbc: bool
    d_nn_A: float
    initial_atoms: Any
    final_atoms: Any

    def to_row(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "atom_index0": self.atom_index,
            "atom_index1": self.atom_index + 1,
            "atom_symbol": self.atom_symbol,
            "vacancy_site_index0": self.vacancy.site_index0,
            "vacancy_source_index": self.vacancy.source_index,
            "hop_distance_A": self.hop_distance_A,
            "direct_distance_A": self.direct_distance_A,
            "mic_distance_A": self.mic_distance_A,
            "crosses_pbc": self.crosses_pbc,
            "d_nn_A": self.d_nn_A,
            "vacancy_x_A": float(self.vacancy.cartesian[0]),
            "vacancy_y_A": float(self.vacancy.cartesian[1]),
            "vacancy_z_A": float(self.vacancy.cartesian[2]),
            "new_vacancy_x_A": float(self.new_vacancy_cartesian[0]),
            "new_vacancy_y_A": float(self.new_vacancy_cartesian[1]),
            "new_vacancy_z_A": float(self.new_vacancy_cartesian[2]),
        }


def minimum_image(delta: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    if not np.any(pbc):
        return delta
    inv = np.linalg.inv(cell)
    frac = delta @ inv
    frac[pbc] -= np.round(frac[pbc])
    return frac @ cell


def estimate_nn_from_atoms(atoms: Any) -> float:
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    distances: list[float] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta = minimum_image(positions[j] - positions[i], cell, pbc)
            distance = float(np.linalg.norm(delta))
            if distance > 0.1:
                distances.append(distance)
    if not distances:
        raise ValueError("cannot estimate nearest-neighbor distance from fewer than two atoms")
    ordered = np.sort(np.asarray(distances, dtype=float))
    low_shell = ordered[: max(6, min(len(ordered), len(positions) * 3))]
    return float(np.median(low_shell))


def _event_id(step: int, atom_index: int, symbol: str, old_pos: np.ndarray, vacancy_pos: np.ndarray, cell: np.ndarray) -> str:
    payload = {
        "step": step,
        "atom_index": atom_index,
        "symbol": symbol,
        "old_pos": np.round(old_pos, 4).tolist(),
        "vacancy": np.round(vacancy_pos, 4).tolist(),
        "cell": np.round(cell, 4).tolist(),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"step{step:04d}_atom{atom_index:04d}_{symbol}_{digest}"


def build_final_atoms_for_hop(atoms: Any, atom_index: int, vacancy_cartesian: np.ndarray) -> Any:
    final_atoms = atoms.copy()
    positions = np.asarray(final_atoms.get_positions(), dtype=float)
    positions[atom_index] = vacancy_cartesian
    final_atoms.set_positions(positions)
    return final_atoms


def generate_hop_events(
    atoms: Any,
    vacancy: VacancySite,
    step: int,
    d_nn_A: float,
    shell_low: float,
    shell_high: float,
    allow_pbc_hop: bool,
    pbc_cross_tol_A: float,
    max_events: int | None = None,
) -> list[HopEvent]:
    """Generate atom-to-vacancy first-neighbor hop events."""

    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    vacancy_pos = np.asarray(vacancy.cartesian, dtype=float)

    lo = shell_low * d_nn_A
    hi = shell_high * d_nn_A
    events: list[HopEvent] = []

    for atom_index, old_pos in enumerate(positions):
        direct_vec = vacancy_pos - old_pos
        mic_vec = minimum_image(direct_vec, cell, pbc)
        direct_distance = float(np.linalg.norm(direct_vec))
        mic_distance = float(np.linalg.norm(mic_vec))
        crosses_pbc = bool(abs(direct_distance - mic_distance) > pbc_cross_tol_A)
        if crosses_pbc and not allow_pbc_hop:
            continue
        if mic_distance < lo or mic_distance > hi:
            continue

        final_pos = old_pos + mic_vec
        final_atoms = build_final_atoms_for_hop(atoms, atom_index, final_pos)
        symbol = atoms[atom_index].symbol
        events.append(
            HopEvent(
                event_id=_event_id(step, atom_index, symbol, old_pos, vacancy_pos, cell),
                atom_index=atom_index,
                atom_symbol=symbol,
                vacancy=vacancy,
                atom_initial_cartesian=old_pos.copy(),
                atom_final_cartesian=final_pos.copy(),
                new_vacancy_cartesian=old_pos.copy(),
                hop_distance_A=mic_distance,
                direct_distance_A=direct_distance,
                mic_distance_A=mic_distance,
                crosses_pbc=crosses_pbc,
                d_nn_A=d_nn_A,
                initial_atoms=atoms.copy(),
                final_atoms=final_atoms,
            )
        )

    events.sort(key=lambda event: (event.hop_distance_A, event.atom_index))
    if max_events is not None:
        events = events[:max_events]
    return events

