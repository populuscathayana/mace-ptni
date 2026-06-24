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
    event_type: str
    atom_index: int
    atom_symbol: str
    vacancy: VacancySite
    initial_coordination: int
    final_coordination: int
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
            "event_type": self.event_type,
            "atom_index0": self.atom_index,
            "atom_index1": self.atom_index + 1,
            "atom_symbol": self.atom_symbol,
            "initial_coordination": self.initial_coordination,
            "final_coordination_at_target": self.final_coordination,
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


def coordination_numbers(atoms: Any, d_nn_A: float, cutoff_factor: float) -> np.ndarray:
    """Count neighbors within a close-packed shell cutoff for each atom."""

    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    cutoff = cutoff_factor * d_nn_A
    counts = np.zeros(len(positions), dtype=int)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta = minimum_image(positions[j] - positions[i], cell, pbc)
            distance = float(np.linalg.norm(delta))
            if 0.1 < distance <= cutoff:
                counts[i] += 1
                counts[j] += 1
    return counts


def coordination_at_position(
    atoms: Any,
    position: np.ndarray,
    exclude_index: int,
    d_nn_A: float,
    cutoff_factor: float,
) -> int:
    """Count real-atom neighbors around a trial position."""

    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    cutoff = cutoff_factor * d_nn_A
    count = 0
    for index, atom_pos in enumerate(positions):
        if index == exclude_index:
            continue
        delta = minimum_image(atom_pos - position, cell, pbc)
        distance = float(np.linalg.norm(delta))
        if 0.1 < distance <= cutoff:
            count += 1
    return count


def _event_id(
    step: int,
    atom_index: int,
    symbol: str,
    old_pos: np.ndarray,
    vacancy_pos: np.ndarray,
    cell: np.ndarray,
    event_type: str,
) -> str:
    payload = {
        "step": step,
        "event_type": event_type,
        "atom_index": atom_index,
        "symbol": symbol,
        "old_pos": np.round(old_pos, 4).tolist(),
        "vacancy": np.round(vacancy_pos, 4).tolist(),
        "cell": np.round(cell, 4).tolist(),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"step{step:04d}_{event_type}_atom{atom_index:04d}_{symbol}_{digest}"


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
    """Generate atom-to-vacancy first-neighbor hop events around one vacancy."""

    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    vacancy_pos = np.asarray(vacancy.cartesian, dtype=float)
    coord = coordination_numbers(atoms, d_nn_A, shell_high)

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

        symbol = atoms[atom_index].symbol
        final_pos = old_pos + mic_vec
        final_coord = coordination_at_position(atoms, final_pos, atom_index, d_nn_A, shell_high)
        event_type = classify_event_type(
            symbol,
            int(coord[atom_index]),
            final_coord,
            ni_dissolve_initial_max=8,
            ni_dissolve_final_max=2,
        )
        final_atoms = build_final_atoms_for_hop(atoms, atom_index, final_pos)
        events.append(
            HopEvent(
                event_id=_event_id(step, atom_index, symbol, old_pos, vacancy_pos, cell, event_type),
                event_type=event_type,
                atom_index=atom_index,
                atom_symbol=symbol,
                vacancy=vacancy,
                initial_coordination=int(coord[atom_index]),
                final_coordination=final_coord,
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


def classify_event_type(
    symbol: str,
    initial_coordination: int,
    final_coordination: int,
    ni_dissolve_initial_max: int,
    ni_dissolve_final_max: int,
) -> str:
    if (
        symbol == "Ni"
        and initial_coordination <= ni_dissolve_initial_max
        and final_coordination <= ni_dissolve_final_max
    ):
        return "dissolution"
    return "hop"


def generate_atom_centric_hop_events(
    atoms: Any,
    sites: list[VacancySite],
    step: int,
    d_nn_A: float,
    shell_low: float,
    shell_high: float,
    coord_cutoff_factor: float,
    mobile_coordination_max: int,
    ni_dissolve_initial_max: int,
    ni_dissolve_final_max: int,
    allow_pbc_hop: bool,
    pbc_cross_tol_A: float,
    rng: np.random.Generator,
) -> tuple[list[HopEvent], dict[str, Any]]:
    """Pick one random under-coordinated atom, then return all legal neighboring sites."""

    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    coord = coordination_numbers(atoms, d_nn_A, coord_cutoff_factor)
    mobile_indices = np.where(coord < mobile_coordination_max)[0].astype(int).tolist()
    rng.shuffle(mobile_indices)

    lo = shell_low * d_nn_A
    hi = shell_high * d_nn_A
    checked_atoms = 0
    rejected_without_sites = 0

    for atom_index in mobile_indices:
        checked_atoms += 1
        old_pos = positions[atom_index]
        legal: list[tuple[VacancySite, np.ndarray, float, float, bool, int, str]] = []
        for site in sites:
            vacancy_pos = np.asarray(site.cartesian, dtype=float)
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
            final_coord = coordination_at_position(atoms, final_pos, atom_index, d_nn_A, coord_cutoff_factor)
            symbol = atoms[atom_index].symbol
            event_type = classify_event_type(
                symbol,
                int(coord[atom_index]),
                final_coord,
                ni_dissolve_initial_max,
                ni_dissolve_final_max,
            )
            legal.append((site, final_pos, direct_distance, mic_distance, crosses_pbc, final_coord, event_type))

        if not legal:
            rejected_without_sites += 1
            continue

        symbol = atoms[atom_index].symbol
        events: list[HopEvent] = []
        for site, final_pos, direct_distance, mic_distance, crosses_pbc, final_coord, event_type in legal:
            final_atoms = build_final_atoms_for_hop(atoms, atom_index, final_pos)
            events.append(
                HopEvent(
                    event_id=_event_id(
                        step,
                        atom_index,
                        symbol,
                        old_pos,
                        np.asarray(site.cartesian, dtype=float),
                        cell,
                        event_type,
                    ),
                    event_type=event_type,
                    atom_index=atom_index,
                    atom_symbol=symbol,
                    vacancy=site,
                    initial_coordination=int(coord[atom_index]),
                    final_coordination=final_coord,
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
        diagnostics = {
            "selection_mode": "atom_random",
            "mobile_atom_count": len(mobile_indices),
            "checked_mobile_atoms": checked_atoms,
            "mobile_atoms_without_legal_site": rejected_without_sites,
            "selected_atom_legal_site_count": len(legal),
            "selected_atom_coordination": int(coord[atom_index]),
            "selected_atom_final_coordination": ",".join(str(item[5]) for item in legal),
        }
        return events, diagnostics

    return [], {
        "selection_mode": "atom_random",
        "mobile_atom_count": len(mobile_indices),
        "checked_mobile_atoms": checked_atoms,
        "mobile_atoms_without_legal_site": rejected_without_sites,
        "selected_atom_legal_site_count": 0,
        "selected_atom_coordination": "",
        "selected_atom_final_coordination": "",
    }
