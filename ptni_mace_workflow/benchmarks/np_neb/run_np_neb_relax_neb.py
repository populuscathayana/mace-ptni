#!/usr/bin/env python
"""Run MACE endpoint relaxation + CI-NEB for NP 00/01/02 triplets.

Input is the NP benchmark extxyz package with one final DFT frame per OUTCAR.
For each complete triplet:

1. Evaluate MACE single-point energies on the DFT 00/01/02 structures.
2. Relax 00 and 02 endpoints with MACE.
3. Build a NEB path between relaxed endpoints, using the DFT 01 image as the
   central initial image when possible.
4. Repair possible NEB insertion/path discontinuity if needed.
5. Run MACE NEB and compare the resulting barrier to the DFT 00/01/02 barrier.

Important output behavior:
- If path repair is applied, the repaired path is written even without --write-images.
- If --write-images is enabled, both raw and repaired/final paths are written.
- RMSD diagnostics compare optimized IS/TS/FS against the original DFT/reordered
  00/01/02 references, not against the fixed NEB endpoints themselves.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def safe_label(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "group"


def get_float_info(atoms: Any, key: str) -> float | None:
    value = atoms.info.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def get_int_info(atoms: Any, key: str) -> int | None:
    value = atoms.info.get(key)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def max_force(atoms: Any) -> float:
    import numpy as np

    forces = atoms.get_forces()
    return float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0


def rmsd_raw(reference: Any, candidate: Any) -> float:
    import numpy as np

    if len(reference) != len(candidate):
        raise ValueError("RMSD requires same atom count")
    ref_pos = reference.get_positions()
    cand_pos = candidate.get_positions()
    diff = cand_pos - ref_pos
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def kabsch_displacements(reference: Any, candidate: Any) -> Any:
    import numpy as np

    if len(reference) != len(candidate):
        raise ValueError("RMSD requires same atom count")
    ref_pos = reference.get_positions()
    cand_pos = candidate.get_positions()

    ref_centered = ref_pos - ref_pos.mean(axis=0)
    cand_centered = cand_pos - cand_pos.mean(axis=0)

    covariance = cand_centered.T @ ref_centered
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    rotation = vt_mat.T @ u_mat.T

    if np.linalg.det(rotation) < 0:
        vt_mat[-1, :] *= -1.0
        rotation = vt_mat.T @ u_mat.T

    aligned = cand_centered @ rotation
    diff = aligned - ref_centered
    return np.linalg.norm(diff, axis=1)


def rmsd_kabsch(reference: Any, candidate: Any) -> float:
    import numpy as np

    displacements = kabsch_displacements(reference, candidate)
    return float(np.sqrt(np.mean(displacements * displacements)))


def rmsd_pair(reference: Any, candidate: Any) -> dict[str, Any]:
    import numpy as np

    if reference.get_chemical_symbols() != candidate.get_chemical_symbols():
        raise ValueError("RMSD requires identical atom order and symbols")

    aligned_displacements = kabsch_displacements(reference, candidate)
    max_index = int(np.argmax(aligned_displacements)) if len(aligned_displacements) else -1
    symbols = reference.get_chemical_symbols()

    return {
        "raw_A": rmsd_raw(reference, candidate),
        "aligned_A": float(np.sqrt(np.mean(aligned_displacements * aligned_displacements))) if len(aligned_displacements) else 0.0,
        "max_atom_aligned_A": float(aligned_displacements[max_index]) if max_index >= 0 else 0.0,
        "max_atom_index": max_index,
        "max_atom_element": symbols[max_index] if max_index >= 0 else "",
    }


def species_preserving_reorder(reference: Any, atoms: Any, use_pbc: bool = False) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("scipy is required for --reorder-path-atoms") from exc

    if len(reference) != len(atoms):
        raise ValueError("reorder requires same atom count")

    ref_symbols = np.array(reference.get_chemical_symbols())
    atom_symbols = np.array(atoms.get_chemical_symbols())

    if sorted(ref_symbols.tolist()) != sorted(atom_symbols.tolist()):
        raise ValueError("reorder requires same composition")

    mapping = np.arange(len(reference), dtype=int)
    ref_pos = reference.get_positions()
    atom_pos = atoms.get_positions()

    for symbol in sorted(set(ref_symbols.tolist())):
        ref_idx = np.where(ref_symbols == symbol)[0]
        atom_idx = np.where(atom_symbols == symbol)[0]

        if use_pbc:
            ref_frac = reference.get_scaled_positions(wrap=False)[ref_idx]
            atom_frac = atoms.get_scaled_positions(wrap=False)[atom_idx]
            cost = np.zeros((len(ref_idx), len(atom_idx)), dtype=float)

            for row, frac in enumerate(ref_frac):
                dfrac = atom_frac - frac[None, :]
                dfrac -= np.round(dfrac)
                dcart = dfrac @ reference.cell.array
                cost[row, :] = np.sum(dcart * dcart, axis=1)
        else:
            diff = ref_pos[ref_idx][:, None, :] - atom_pos[atom_idx][None, :, :]
            cost = np.sum(diff * diff, axis=2)

        rows, cols = linear_sum_assignment(cost)
        mapping[ref_idx[rows]] = atom_idx[cols]

    original_disp = np.linalg.norm(atom_pos - ref_pos, axis=1)

    reordered = atoms[mapping].copy()
    reordered.info.update(atoms.info)

    reordered_disp = np.linalg.norm(reordered.get_positions() - ref_pos, axis=1)

    changed = int(np.sum(mapping != np.arange(len(mapping))))
    original_rmse = float(np.sqrt(np.mean(original_disp * original_disp))) if len(original_disp) else 0.0
    reordered_rmse = float(np.sqrt(np.mean(reordered_disp * reordered_disp))) if len(reordered_disp) else 0.0
    gain = 0.0 if original_rmse <= 1e-12 else (original_rmse - reordered_rmse) / original_rmse

    return reordered, {
        "changed_count": changed,
        "original_atom_rmsd_A": original_rmse,
        "reordered_atom_rmsd_A": reordered_rmse,
        "gain_fraction": gain,
        "max_original_displacement_A": float(original_disp.max()) if len(original_disp) else 0.0,
        "max_reordered_displacement_A": float(reordered_disp.max()) if len(reordered_disp) else 0.0,
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


def apply_pbc_mode(atoms: Any, mode: str) -> None:
    if mode == "true":
        atoms.pbc = (True, True, True)
    elif mode == "false":
        atoms.pbc = (False, False, False)


def prepare_np_geometry(atoms: Any, wrap_scaled: bool, pbc_mode: str) -> Any:
    import numpy as np

    prepared = atoms.copy()

    if wrap_scaled:
        scaled = prepared.get_scaled_positions(wrap=False)
        prepared.set_scaled_positions(np.mod(scaled, 1.0))

    apply_pbc_mode(prepared, pbc_mode)
    return prepared


def interpolate_atoms(a0: Any, a1: Any, t: float) -> Any:
    atoms = a0.copy()
    positions = (1.0 - t) * a0.get_positions() + t * a1.get_positions()
    atoms.set_positions(positions)
    atoms.pbc = a0.pbc
    return atoms


def build_images(is_atoms: Any, ts_atoms: Any, fs_atoms: Any, n_images: int, seed_middle: str) -> list[Any]:
    if n_images < 3:
        raise ValueError("--n-images must be >= 3")
    if n_images % 2 == 0:
        raise ValueError("--n-images must be odd so the TS can be the central image")

    center = n_images // 2
    images = []

    for i in range(n_images):
        if i == 0:
            images.append(is_atoms.copy())
        elif i == n_images - 1:
            images.append(fs_atoms.copy())
        elif seed_middle == "dft_ts" and i == center:
            images.append(ts_atoms.copy())
        elif seed_middle == "dft_ts" and i < center:
            images.append(interpolate_atoms(is_atoms, ts_atoms, i / center))
        elif seed_middle == "dft_ts":
            images.append(interpolate_atoms(ts_atoms, fs_atoms, (i - center) / center))
        else:
            images.append(interpolate_atoms(is_atoms, fs_atoms, i / (n_images - 1)))

    return images


def path_adjacent_diagnostics(images: list[Any]) -> list[dict[str, Any]]:
    import numpy as np

    diagnostics = []

    for index in range(len(images) - 1):
        a0 = images[index]
        a1 = images[index + 1]

        if len(a0) != len(a1):
            raise ValueError("Path diagnostic requires same atom count for adjacent images")
        if a0.get_chemical_symbols() != a1.get_chemical_symbols():
            raise ValueError("Path diagnostic requires same atom order and symbols for adjacent images")

        disp = a1.get_positions() - a0.get_positions()
        norm = np.linalg.norm(disp, axis=1)
        max_index = int(np.argmax(norm)) if len(norm) else -1
        symbols = a0.get_chemical_symbols()

        diagnostics.append(
            {
                "segment": f"{index}->{index + 1}",
                "max_atom_step_A": float(norm[max_index]) if max_index >= 0 else 0.0,
                "rms_atom_step_A": float(np.sqrt(np.mean(norm * norm))) if len(norm) else 0.0,
                "mean_atom_step_A": float(np.mean(norm)) if len(norm) else 0.0,
                "max_atom_index": max_index,
                "max_atom_element": symbols[max_index] if max_index >= 0 else "",
            }
        )

    return diagnostics


def summarize_path_diagnostics(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    if not diagnostics:
        return {
            "max_step_A": 0.0,
            "rms_step_A": 0.0,
            "max_segment": "",
            "max_atom_index": -1,
            "max_atom_element": "",
        }

    max_item = max(diagnostics, key=lambda item: item["max_atom_step_A"])
    rms_values = [float(item["rms_atom_step_A"]) for item in diagnostics]
    rms_step = math.sqrt(sum(value * value for value in rms_values) / len(rms_values)) if rms_values else 0.0

    return {
        "max_step_A": float(max_item["max_atom_step_A"]),
        "rms_step_A": float(rms_step),
        "max_segment": max_item["segment"],
        "max_atom_index": int(max_item["max_atom_index"]),
        "max_atom_element": max_item["max_atom_element"],
    }


def unwrap_atoms_to_reference(reference: Any, atoms: Any) -> tuple[Any, dict[str, Any]]:
    """Shift each atom of atoms by integer cell vectors to be closest to reference.

    This repairs artificial long jumps caused by cell-boundary wrapping.
    It is conservative and only applies integer-cell shifts in fractional space.
    """
    import numpy as np

    if len(reference) != len(atoms):
        raise ValueError("unwrap requires same atom count")
    if reference.get_chemical_symbols() != atoms.get_chemical_symbols():
        raise ValueError("unwrap requires identical atom order and symbols")

    cell_volume = abs(float(reference.cell.volume))
    if cell_volume <= 1e-12:
        return atoms.copy(), {
            "shifted_atom_count": 0,
            "max_shift_norm": 0.0,
            "before_max_atom_step_A": 0.0,
            "after_max_atom_step_A": 0.0,
            "message": "cell volume is zero; unwrap skipped",
        }

    ref_scaled = reference.get_scaled_positions(wrap=False)
    atom_scaled = atoms.get_scaled_positions(wrap=False)

    dfrac = atom_scaled - ref_scaled
    shifts = np.round(dfrac)
    repaired_scaled = atom_scaled - shifts

    repaired = atoms.copy()
    repaired.set_scaled_positions(repaired_scaled)
    repaired.pbc = atoms.pbc
    repaired.info.update(atoms.info)

    before_disp = atoms.get_positions() - reference.get_positions()
    after_disp = repaired.get_positions() - reference.get_positions()

    before_norm = np.linalg.norm(before_disp, axis=1)
    after_norm = np.linalg.norm(after_disp, axis=1)
    shift_norm = np.linalg.norm(shifts, axis=1)

    shifted_atom_count = int(np.count_nonzero(shift_norm > 1e-12))

    return repaired, {
        "shifted_atom_count": shifted_atom_count,
        "max_shift_norm": float(shift_norm.max()) if len(shift_norm) else 0.0,
        "before_max_atom_step_A": float(before_norm.max()) if len(before_norm) else 0.0,
        "after_max_atom_step_A": float(after_norm.max()) if len(after_norm) else 0.0,
        "message": "minimum-image anchor unwrap",
    }


def annotate_neb_images(images: list[Any], repair_info: dict[str, Any]) -> None:
    for index, image in enumerate(images):
        image.info["neb_image_index"] = index
        image.info["neb_path_repair_enabled"] = int(bool(repair_info.get("enabled", False)))
        image.info["neb_path_repair_applied"] = int(bool(repair_info.get("applied", False)))
        image.info["neb_path_repair_message"] = str(repair_info.get("message", ""))
        image.info["neb_path_raw_max_step_A"] = float(repair_info.get("raw_max_step_A", 0.0))
        image.info["neb_path_final_max_step_A"] = float(repair_info.get("final_max_step_A", 0.0))


def repair_neb_images_if_needed(images: list[Any], enabled: bool) -> tuple[list[Any], dict[str, Any]]:
    """Repair NEB path insertion discontinuity and return the actual path to use.

    The repair:
    - keeps image 0 as IS anchor;
    - unwraps center image relative to image 0;
    - unwraps final image relative to the repaired center image;
    - rebuilds the odd-image path through the repaired anchors.

    If no integer-cell shift is found and the path max step is not reduced,
    the original path is returned.
    """
    if len(images) < 3:
        raise ValueError("NEB path repair requires at least 3 images")
    if len(images) % 2 == 0:
        raise ValueError("NEB path repair assumes an odd number of images")

    raw_diag = path_adjacent_diagnostics(images)
    raw_summary = summarize_path_diagnostics(raw_diag)

    base_info = {
        "enabled": enabled,
        "applied": False,
        "message": "repair disabled" if not enabled else "no repair needed",
        "raw_max_step_A": raw_summary["max_step_A"],
        "raw_rms_step_A": raw_summary["rms_step_A"],
        "raw_max_segment": raw_summary["max_segment"],
        "raw_max_atom_index": raw_summary["max_atom_index"],
        "raw_max_atom_element": raw_summary["max_atom_element"],
        "final_max_step_A": raw_summary["max_step_A"],
        "final_rms_step_A": raw_summary["rms_step_A"],
        "final_max_segment": raw_summary["max_segment"],
        "final_max_atom_index": raw_summary["max_atom_index"],
        "final_max_atom_element": raw_summary["max_atom_element"],
        "center_shifted_atom_count": 0,
        "final_shifted_atom_count": 0,
    }

    if not enabled:
        annotate_neb_images(images, base_info)
        return images, base_info

    n_images = len(images)
    center = n_images // 2

    is_anchor = images[0].copy()
    ts_anchor, ts_unwrap = unwrap_atoms_to_reference(is_anchor, images[center])
    fs_anchor, fs_unwrap = unwrap_atoms_to_reference(ts_anchor, images[-1])

    repaired = []

    for i in range(n_images):
        if i == 0:
            repaired.append(is_anchor.copy())
        elif i == center:
            repaired.append(ts_anchor.copy())
        elif i == n_images - 1:
            repaired.append(fs_anchor.copy())
        elif i < center:
            repaired.append(interpolate_atoms(is_anchor, ts_anchor, i / center))
        else:
            denominator = (n_images - 1) - center
            repaired.append(interpolate_atoms(ts_anchor, fs_anchor, (i - center) / denominator))

    final_diag = path_adjacent_diagnostics(repaired)
    final_summary = summarize_path_diagnostics(final_diag)

    shifted_count = int(ts_unwrap["shifted_atom_count"]) + int(fs_unwrap["shifted_atom_count"])
    max_step_reduced = final_summary["max_step_A"] < raw_summary["max_step_A"] - 1e-8
    applied = shifted_count > 0 or max_step_reduced

    if not applied:
        annotate_neb_images(images, base_info)
        return images, base_info

    info = {
        "enabled": True,
        "applied": True,
        "message": (
            "minimum-image anchor repair applied; "
            f"center_shifted_atoms={ts_unwrap['shifted_atom_count']}; "
            f"final_shifted_atoms={fs_unwrap['shifted_atom_count']}; "
            f"max_step_A={raw_summary['max_step_A']:.6f}->{final_summary['max_step_A']:.6f}"
        ),
        "raw_max_step_A": raw_summary["max_step_A"],
        "raw_rms_step_A": raw_summary["rms_step_A"],
        "raw_max_segment": raw_summary["max_segment"],
        "raw_max_atom_index": raw_summary["max_atom_index"],
        "raw_max_atom_element": raw_summary["max_atom_element"],
        "final_max_step_A": final_summary["max_step_A"],
        "final_rms_step_A": final_summary["rms_step_A"],
        "final_max_segment": final_summary["max_segment"],
        "final_max_atom_index": final_summary["max_atom_index"],
        "final_max_atom_element": final_summary["max_atom_element"],
        "center_shifted_atom_count": int(ts_unwrap["shifted_atom_count"]),
        "final_shifted_atom_count": int(fs_unwrap["shifted_atom_count"]),
    }

    annotate_neb_images(repaired, info)
    return repaired, info


def relax_atoms(
    atoms: Any,
    calc: Any,
    fmax: float,
    steps: int,
    trajectory: Path | None,
    maxstep: float | None,
    downhill_check: bool,
) -> tuple[Any, float, float, bool]:
    from ase.optimize import FIRE

    atoms = atoms.copy()
    attach_calc(atoms, calc)

    try:
        opt = FIRE(
            atoms,
            trajectory=str(trajectory) if trajectory else None,
            logfile=None,
            maxstep=maxstep,
            downhill_check=downhill_check,
        )
        converged = bool(opt.run(fmax=fmax, steps=steps))
        energy = float(atoms.get_potential_energy())
        force = max_force(atoms)
        return atoms, energy, force, converged
    finally:
        detach_calc(atoms)


def run_neb(
    images: list[Any],
    calc: Any,
    fmax: float,
    steps: int,
    climb: bool,
    trajectory: Path | None,
    maxstep: float | None,
    downhill_check: bool,
) -> tuple[list[float], list[float], bool]:
    from ase.mep import NEB
    from ase.optimize import FIRE

    attach_calc(images, calc)

    try:
        neb = NEB(images, climb=climb, allow_shared_calculator=True, method="improvedtangent")
        opt = FIRE(
            neb,
            trajectory=str(trajectory) if trajectory else None,
            logfile=None,
            maxstep=maxstep,
            downhill_check=downhill_check,
        )
        converged = bool(opt.run(fmax=fmax, steps=steps))
        energies = [float(image.get_potential_energy()) for image in images]
        forces = [max_force(image) for image in images]
        return energies, forces, converged
    finally:
        detach_calc(images)


def read_triplets(
    configs: Path,
    group_key: str,
    image_key: str,
    wrap_scaled: bool,
    pbc_mode: str,
) -> dict[str, dict[int, Any]]:
    from ase.io import iread

    groups: dict[str, dict[int, Any]] = defaultdict(dict)

    for atoms in iread(configs.as_posix(), index=":"):
        group = atoms.info.get(group_key)
        image = get_int_info(atoms, image_key)

        if not group or image not in {0, 1, 2}:
            continue

        groups[str(group)][int(image)] = prepare_np_geometry(atoms, wrap_scaled, pbc_mode)

    return groups


def check_triplet(group: str, by_image: dict[int, Any]) -> tuple[Any, Any, Any]:
    if set(by_image) != {0, 1, 2}:
        raise ValueError(f"{group}: missing one of 00/01/02")

    is_atoms = by_image[0]
    ts_atoms = by_image[1]
    fs_atoms = by_image[2]

    if len({len(is_atoms), len(ts_atoms), len(fs_atoms)}) != 1:
        raise ValueError(f"{group}: atom counts differ")

    if (
        is_atoms.get_chemical_symbols() != ts_atoms.get_chemical_symbols()
        or is_atoms.get_chemical_symbols() != fs_atoms.get_chemical_symbols()
    ):
        raise ValueError(f"{group}: chemical symbol order differs")

    return is_atoms, ts_atoms, fs_atoms


def singlepoint_triplet(is_atoms: Any, ts_atoms: Any, fs_atoms: Any, calc: Any) -> tuple[list[float], list[float]]:
    images = [is_atoms.copy(), ts_atoms.copy(), fs_atoms.copy()]
    attach_calc(images, calc)

    try:
        energies = [float(image.get_potential_energy()) for image in images]
        forces = [max_force(image) for image in images]
        return energies, forces
    finally:
        detach_calc(images)


def dft_triplet_energies(is_atoms: Any, ts_atoms: Any, fs_atoms: Any, key: str) -> list[float]:
    values = [get_float_info(atoms, key) for atoms in (is_atoms, ts_atoms, fs_atoms)]

    if any(value is None for value in values):
        raise ValueError(f"missing {key} on one of 00/01/02")

    return [float(value) for value in values]


def write_images(path: Path, images: list[Any]) -> None:
    from ase.io import write

    path.parent.mkdir(parents=True, exist_ok=True)
    write(path.as_posix(), images, format="extxyz")


def write_optimized_triplet(path: Path, images: list[Any], rmsd_values: dict[str, float], ts_index: int) -> None:
    from ase.io import write

    roles = [
        ("is", "00", images[0]),
        ("ts", "01", images[ts_index]),
        ("fs", "02", images[-1]),
    ]

    out_images = []

    for role, ref_image, atoms in roles:
        image = atoms.copy()
        image.info["optimized_role"] = role
        image.info["reference_image"] = ref_image
        image.info["neb_ts_image_index"] = ts_index
        image.info[f"rmsd_{role}_raw_A"] = rmsd_values[f"rmsd_{role}_raw_A"]
        image.info[f"rmsd_{role}_aligned_A"] = rmsd_values[f"rmsd_{role}_aligned_A"]
        image.info[f"max_atom_rmsd_{role}_aligned_A"] = rmsd_values[f"max_atom_rmsd_{role}_aligned_A"]
        image.info[f"max_atom_rmsd_{role}_index"] = rmsd_values[f"max_atom_rmsd_{role}_index"]
        image.info[f"max_atom_rmsd_{role}_element"] = rmsd_values[f"max_atom_rmsd_{role}_element"]
        out_images.append(image)

    path.parent.mkdir(parents=True, exist_ok=True)
    write(path.as_posix(), out_images, format="extxyz")


def metric_triplet(energies: list[float]) -> dict[str, float]:
    return {
        "forward_barrier_eV": energies[1] - energies[0],
        "reverse_barrier_eV": energies[1] - energies[2],
        "lower_endpoint_barrier_eV": energies[1] - min(energies[0], energies[2]),
        "reaction_energy_eV": energies[2] - energies[0],
    }


def process_group(group: str, by_image: dict[int, Any], calc: Any, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    group_dir = out_dir / "groups" / safe_label(group)
    group_dir.mkdir(parents=True, exist_ok=True)

    is_atoms, ts_atoms, fs_atoms = check_triplet(group, by_image)

    path_is_atoms = is_atoms.copy()

    if args.reorder_path_atoms:
        path_ts_atoms, ts_reorder = species_preserving_reorder(
            path_is_atoms,
            ts_atoms,
            use_pbc=args.reorder_metric == "pbc",
        )
        path_fs_atoms, fs_reorder = species_preserving_reorder(
            path_ts_atoms,
            fs_atoms,
            use_pbc=args.reorder_metric == "pbc",
        )
    else:
        path_ts_atoms = ts_atoms.copy()
        path_fs_atoms = fs_atoms.copy()
        ts_reorder = {
            "changed_count": 0,
            "original_atom_rmsd_A": 0.0,
            "reordered_atom_rmsd_A": 0.0,
            "gain_fraction": 0.0,
            "max_original_displacement_A": 0.0,
            "max_reordered_displacement_A": 0.0,
        }
        fs_reorder = dict(ts_reorder)

    dft_energies = dft_triplet_energies(is_atoms, ts_atoms, fs_atoms, args.ref_energy_key)
    mace_sp_energies, mace_sp_forces = singlepoint_triplet(is_atoms, ts_atoms, fs_atoms, calc)

    is_relaxed, e_is_relaxed, f_is_relaxed, is_converged = relax_atoms(
        path_is_atoms,
        calc,
        args.endpoint_fmax,
        args.endpoint_steps,
        group_dir / "is_relax.traj" if args.write_trajectories else None,
        args.endpoint_maxstep,
        args.fire_downhill_check,
    )

    fs_relaxed, e_fs_relaxed, f_fs_relaxed, fs_converged = relax_atoms(
        path_fs_atoms,
        calc,
        args.endpoint_fmax,
        args.endpoint_steps,
        group_dir / "fs_relax.traj" if args.write_trajectories else None,
        args.endpoint_maxstep,
        args.fire_downhill_check,
    )

    images_raw = build_images(is_relaxed, path_ts_atoms, fs_relaxed, args.n_images, args.seed_middle)

    if args.write_images:
        write_images(group_dir / "neb_initial_raw.extxyz", images_raw)

    images, repair_info = repair_neb_images_if_needed(images_raw, enabled=args.repair_neb_path)

    if repair_info["applied"]:
        write_images(group_dir / "neb_initial.extxyz", images)
        write_images(group_dir / "neb_initial_repaired.extxyz", images)
    elif args.write_images:
        write_images(group_dir / "neb_initial.extxyz", images)

    neb_energies, neb_forces, neb_converged = run_neb(
        images,
        calc,
        args.neb_fmax,
        args.neb_steps,
        climb=not args.no_climb,
        trajectory=group_dir / "neb.traj" if args.write_trajectories else None,
        maxstep=args.neb_maxstep,
        downhill_check=args.fire_downhill_check,
    )

    if args.write_images or repair_info["applied"]:
        write_images(group_dir / "neb_final.extxyz", images)

    dft_metrics = metric_triplet(dft_energies)
    mace_sp_metrics = metric_triplet(mace_sp_energies)

    neb_metrics = {
        "forward_barrier_eV": max(neb_energies) - neb_energies[0],
        "reverse_barrier_eV": max(neb_energies) - neb_energies[-1],
        "lower_endpoint_barrier_eV": max(neb_energies) - min(neb_energies[0], neb_energies[-1]),
        "reaction_energy_eV": neb_energies[-1] - neb_energies[0],
    }

    ts_index = neb_energies.index(max(neb_energies))

    # Correct RMSD definition:
    # IS/FS endpoints are fixed during NEB, so comparing them to images[0]/images[-1]
    # before NEB would always give zero. Here we compare the optimized NEB IS/TS/FS
    # against the original DFT/reordered references, as in the original script.
    rmsd_is = rmsd_pair(path_is_atoms, images[0])
    rmsd_ts = rmsd_pair(path_ts_atoms, images[ts_index])
    rmsd_fs = rmsd_pair(path_fs_atoms, images[-1])

    rmsd_values = {
        "rmsd_is_raw_A": rmsd_is["raw_A"],
        "rmsd_is_aligned_A": rmsd_is["aligned_A"],
        "max_atom_rmsd_is_aligned_A": rmsd_is["max_atom_aligned_A"],
        "max_atom_rmsd_is_index": rmsd_is["max_atom_index"],
        "max_atom_rmsd_is_element": rmsd_is["max_atom_element"],
        "rmsd_ts_raw_A": rmsd_ts["raw_A"],
        "rmsd_ts_aligned_A": rmsd_ts["aligned_A"],
        "max_atom_rmsd_ts_aligned_A": rmsd_ts["max_atom_aligned_A"],
        "max_atom_rmsd_ts_index": rmsd_ts["max_atom_index"],
        "max_atom_rmsd_ts_element": rmsd_ts["max_atom_element"],
        "rmsd_fs_raw_A": rmsd_fs["raw_A"],
        "rmsd_fs_aligned_A": rmsd_fs["aligned_A"],
        "max_atom_rmsd_fs_aligned_A": rmsd_fs["max_atom_aligned_A"],
        "max_atom_rmsd_fs_index": rmsd_fs["max_atom_index"],
        "max_atom_rmsd_fs_element": rmsd_fs["max_atom_element"],
    }

    if args.write_images or repair_info["applied"]:
        write_optimized_triplet(group_dir / "optimized_is_ts_fs.extxyz", images, rmsd_values, ts_index)

    row: dict[str, Any] = {
        "neb_group": group,
        "status": "ok",
        "natoms": len(is_atoms),
        "n_images": args.n_images,
        "wrap_scaled": args.wrap_scaled,
        "pbc_mode": args.pbc,
        "interpolation": "cartesian_no_mic",
        "path_atom_reorder": args.reorder_path_atoms,
        "path_atom_reorder_metric": args.reorder_metric,
        "path_reorder_ts_changed_count": ts_reorder["changed_count"],
        "path_reorder_ts_gain_fraction": ts_reorder["gain_fraction"],
        "path_reorder_ts_original_atom_rmsd_A": ts_reorder["original_atom_rmsd_A"],
        "path_reorder_ts_reordered_atom_rmsd_A": ts_reorder["reordered_atom_rmsd_A"],
        "path_reorder_fs_changed_count": fs_reorder["changed_count"],
        "path_reorder_fs_gain_fraction": fs_reorder["gain_fraction"],
        "path_reorder_fs_original_atom_rmsd_A": fs_reorder["original_atom_rmsd_A"],
        "path_reorder_fs_reordered_atom_rmsd_A": fs_reorder["reordered_atom_rmsd_A"],
        "neb_path_repair_enabled": repair_info["enabled"],
        "neb_path_repair_applied": repair_info["applied"],
        "neb_path_repair_message": repair_info["message"],
        "neb_path_raw_max_step_A": repair_info["raw_max_step_A"],
        "neb_path_raw_rms_step_A": repair_info["raw_rms_step_A"],
        "neb_path_raw_max_segment": repair_info["raw_max_segment"],
        "neb_path_raw_max_atom_index": repair_info["raw_max_atom_index"],
        "neb_path_raw_max_atom_element": repair_info["raw_max_atom_element"],
        "neb_path_final_max_step_A": repair_info["final_max_step_A"],
        "neb_path_final_rms_step_A": repair_info["final_rms_step_A"],
        "neb_path_final_max_segment": repair_info["final_max_segment"],
        "neb_path_final_max_atom_index": repair_info["final_max_atom_index"],
        "neb_path_final_max_atom_element": repair_info["final_max_atom_element"],
        "neb_path_center_shifted_atom_count": repair_info["center_shifted_atom_count"],
        "neb_path_final_shifted_atom_count": repair_info["final_shifted_atom_count"],
        "dft_E00_eV": dft_energies[0],
        "dft_E01_eV": dft_energies[1],
        "dft_E02_eV": dft_energies[2],
        "mace_sp_E00_eV": mace_sp_energies[0],
        "mace_sp_E01_eV": mace_sp_energies[1],
        "mace_sp_E02_eV": mace_sp_energies[2],
        "mace_sp_force00_eVA": mace_sp_forces[0],
        "mace_sp_force01_eVA": mace_sp_forces[1],
        "mace_sp_force02_eVA": mace_sp_forces[2],
        "relaxed_E00_eV": e_is_relaxed,
        "relaxed_E02_eV": e_fs_relaxed,
        "relaxed_force00_eVA": f_is_relaxed,
        "relaxed_force02_eVA": f_fs_relaxed,
        "relaxed_00_converged": is_converged,
        "relaxed_02_converged": fs_converged,
        "neb_Emax_eV": max(neb_energies),
        "neb_ts_image_index": ts_index,
        "rmsd_is_to_dft00_raw_A": rmsd_values["rmsd_is_raw_A"],
        "rmsd_is_to_dft00_aligned_A": rmsd_values["rmsd_is_aligned_A"],
        "max_atom_rmsd_is_to_dft00_aligned_A": rmsd_values["max_atom_rmsd_is_aligned_A"],
        "max_atom_rmsd_is_to_dft00_index": rmsd_values["max_atom_rmsd_is_index"],
        "max_atom_rmsd_is_to_dft00_element": rmsd_values["max_atom_rmsd_is_element"],
        "rmsd_ts_to_dft01_raw_A": rmsd_values["rmsd_ts_raw_A"],
        "rmsd_ts_to_dft01_aligned_A": rmsd_values["rmsd_ts_aligned_A"],
        "max_atom_rmsd_ts_to_dft01_aligned_A": rmsd_values["max_atom_rmsd_ts_aligned_A"],
        "max_atom_rmsd_ts_to_dft01_index": rmsd_values["max_atom_rmsd_ts_index"],
        "max_atom_rmsd_ts_to_dft01_element": rmsd_values["max_atom_rmsd_ts_element"],
        "rmsd_fs_to_dft02_raw_A": rmsd_values["rmsd_fs_raw_A"],
        "rmsd_fs_to_dft02_aligned_A": rmsd_values["rmsd_fs_aligned_A"],
        "max_atom_rmsd_fs_to_dft02_aligned_A": rmsd_values["max_atom_rmsd_fs_aligned_A"],
        "max_atom_rmsd_fs_to_dft02_index": rmsd_values["max_atom_rmsd_fs_index"],
        "max_atom_rmsd_fs_to_dft02_element": rmsd_values["max_atom_rmsd_fs_element"],
        "neb_max_force_path_eVA": max(neb_forces),
        "neb_converged": neb_converged,
        "endpoint_maxstep_A": "" if args.endpoint_maxstep is None else args.endpoint_maxstep,
        "neb_maxstep_A": "" if args.neb_maxstep is None else args.neb_maxstep,
        "fire_downhill_check": args.fire_downhill_check,
        "neb_energies_relative_eV": ";".join(f"{energy - neb_energies[0]:.10f}" for energy in neb_energies),
        "neb_forces_max_eVA": ";".join(f"{force:.10f}" for force in neb_forces),
        "group_dir": str(group_dir),
    }

    for prefix, metrics in (("dft", dft_metrics), ("mace_sp", mace_sp_metrics), ("mace_neb", neb_metrics)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value

    for key in dft_metrics:
        row[f"error_sp_{key}"] = mace_sp_metrics[key] - dft_metrics[key]
        row[f"abs_error_sp_{key}"] = abs(row[f"error_sp_{key}"])
        row[f"error_neb_{key}"] = neb_metrics[key] - dft_metrics[key]
        row[f"abs_error_neb_{key}"] = abs(row[f"error_neb_{key}"])

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    seen = set()

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


def stats(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""

    mae = sum(abs(value) for value in values) / len(values)
    rmse = math.sqrt(sum(value * value for value in values) / len(values))

    return mae, rmse


def fmt(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_md(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    repaired_rows = [row for row in ok_rows if row.get("neb_path_repair_applied")]

    lines = [
        "# NP MACE Relaxation + NEB Benchmark",
        "",
        f"- Model: `{args.model.resolve()}`",
        f"- Configs: `{args.configs.resolve()}`",
        f"- Groups attempted: {len(rows)}",
        f"- Groups succeeded: {len(ok_rows)}",
        f"- Groups with NEB path repair: {len(repaired_rows)}",
        f"- Geometry preprocessing: wrap_scaled={args.wrap_scaled}, pbc={args.pbc}",
        f"- Path atom reorder: {args.reorder_path_atoms}, metric={args.reorder_metric}",
        f"- NEB path repair: {args.repair_neb_path}",
        "- NEB interpolation: Cartesian linear interpolation without MIC wrapping after optional path repair.",
        f"- Endpoint relax: fmax={args.endpoint_fmax}, steps={args.endpoint_steps}, maxstep={args.endpoint_maxstep}",
        f"- NEB: fmax={args.neb_fmax}, steps={args.neb_steps}, images={args.n_images}, climb={not args.no_climb}, maxstep={args.neb_maxstep}",
        f"- FIRE downhill check: {args.fire_downhill_check}",
        "",
        "## Error Summary vs DFT 00/01/02 Reference",
        "",
        "| quantity | SP MAE meV | SP RMSE meV | NEB MAE meV | NEB RMSE meV |",
        "|---|---:|---:|---:|---:|",
    ]

    for key, label in [
        ("forward_barrier_eV", "E01 - E00"),
        ("reverse_barrier_eV", "E01 - E02"),
        ("lower_endpoint_barrier_eV", "E01 - min(E00,E02)"),
        ("reaction_energy_eV", "E02 - E00"),
    ]:
        sp_errors = [1000.0 * float(row[f"error_sp_{key}"]) for row in ok_rows if f"error_sp_{key}" in row]
        neb_errors = [1000.0 * float(row[f"error_neb_{key}"]) for row in ok_rows if f"error_neb_{key}" in row]

        sp_mae, sp_rmse = stats(sp_errors)
        neb_mae, neb_rmse = stats(neb_errors)

        lines.append(f"| {label} | {fmt(sp_mae)} | {fmt(sp_rmse)} | {fmt(neb_mae)} | {fmt(neb_rmse)} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- SP uses DFT 00/01/02 geometries directly.",
            "- NEB relaxes 00/02 endpoints with MACE, then runs MACE CI-NEB.",
            "- If NEB path repair is applied, `neb_initial.extxyz` and `neb_initial_repaired.extxyz` contain the repaired path actually used by NEB.",
            "- If `--write-images` is enabled, `neb_initial_raw.extxyz` stores the raw path before repair.",
            "- RMSD diagnostics compare optimized IS/TS/FS against original DFT/reordered 00/01/02 references.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MACE endpoint relaxation and NEB on NP triplets.")

    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", default="float64")
    parser.add_argument("--ref-energy-key", default="REF_energy")
    parser.add_argument("--group-key", default="neb_group")
    parser.add_argument("--image-key", default="neb_image")
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--endpoint-fmax", type=float, default=0.02)
    parser.add_argument("--endpoint-steps", type=int, default=300)
    parser.add_argument(
        "--endpoint-maxstep",
        type=float,
        default=None,
        help="FIRE max atom displacement per endpoint step in A. ASE default is 0.2 A.",
    )
    parser.add_argument("--neb-fmax", type=float, default=0.05)
    parser.add_argument("--neb-steps", type=int, default=400)
    parser.add_argument(
        "--neb-maxstep",
        type=float,
        default=None,
        help="FIRE max atom displacement per NEB step in A. ASE default is 0.2 A.",
    )
    parser.add_argument("--seed-middle", choices=["dft_ts", "interpolate"], default="dft_ts")
    parser.add_argument(
        "--pbc",
        choices=["from-input", "true", "false"],
        default="false",
        help="PBC mode after optional coordinate wrapping. Default false is intended for NP/vacuum structures.",
    )
    parser.add_argument(
        "--no-wrap-scaled",
        dest="wrap_scaled",
        action="store_false",
        help="Do not first map scaled coordinates into [0,1).",
    )
    parser.set_defaults(wrap_scaled=True)

    parser.add_argument(
        "--no-reorder-path-atoms",
        dest="reorder_path_atoms",
        action="store_false",
        help="Disable species-preserving atom reordering before endpoint relaxation/interpolation.",
    )
    parser.set_defaults(reorder_path_atoms=True)

    parser.add_argument(
        "--reorder-metric",
        choices=["no_pbc", "pbc"],
        default="no_pbc",
        help="Distance metric for species-preserving atom reordering.",
    )

    parser.add_argument(
        "--repair-neb-path",
        dest="repair_neb_path",
        action="store_true",
        help="Enable minimum-image anchor repair for NEB path insertion.",
    )
    parser.add_argument(
        "--no-repair-neb-path",
        dest="repair_neb_path",
        action="store_false",
        help="Disable NEB path repair.",
    )
    parser.set_defaults(repair_neb_path=True)

    parser.add_argument("--no-climb", action="store_true")
    parser.add_argument(
        "--fire-downhill-check",
        action="store_true",
        help="Enable FIRE uphill-step rejection where supported.",
    )
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--group-filter", default=None, help="Regex to select NEB groups.")
    parser.add_argument("--write-images", action="store_true")
    parser.add_argument("--write-trajectories", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    args = parser.parse_args()

    if args.n_images < 3 or args.n_images % 2 == 0:
        raise SystemExit("--n-images must be an odd integer >= 3")

    if args.max_groups is not None and args.max_groups < 1:
        raise SystemExit("--max-groups must be >= 1 when provided")

    if not args.configs.is_file():
        raise SystemExit(f"Configs file does not exist: {args.configs}")

    if not args.model.is_file():
        raise SystemExit(f"Model file does not exist: {args.model}")

    from mace.calculators import MACECalculator

    groups = read_triplets(
        args.configs.resolve(),
        args.group_key,
        args.image_key,
        args.wrap_scaled,
        args.pbc,
    )

    selected = []
    group_re = re.compile(args.group_filter) if args.group_filter else None

    for group in sorted(groups):
        if set(groups[group]) != {0, 1, 2}:
            continue
        if group_re and not group_re.search(group):
            continue
        selected.append(group)

    if args.max_groups is not None:
        selected = selected[: args.max_groups]

    if not selected:
        raise SystemExit("No complete 00/01/02 NEB groups selected.")

    calc = MACECalculator(
        model_paths=str(args.model.resolve()),
        device=args.device,
        default_dtype=args.default_dtype,
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for index, group in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {group}")

        try:
            row = process_group(group, groups[group], calc, args, out_dir)
        except Exception as exc:
            if not args.continue_on_error:
                raise

            row = {
                "neb_group": group,
                "status": "failed",
                "message": repr(exc),
            }
            print(f"  failed: {exc!r}")

        rows.append(row)

        if row.get("status") == "ok":
            print(
                "  SP barrier={:.6f} eV; NEB barrier={:.6f} eV; DFT barrier={:.6f} eV".format(
                    row["mace_sp_forward_barrier_eV"],
                    row["mace_neb_forward_barrier_eV"],
                    row["dft_forward_barrier_eV"],
                )
            )
            print(
                "  path repair: applied={}; raw_max_step={:.4f} A; final_max_step={:.4f} A".format(
                    row["neb_path_repair_applied"],
                    row["neb_path_raw_max_step_A"],
                    row["neb_path_final_max_step_A"],
                )
            )
            print(
                "  RMSD raw/aligned/max-atom-aligned A: "
                "IS={:.4f}/{:.4f}/{:.4f}({}{}); "
                "TS={:.4f}/{:.4f}/{:.4f}({}{}); "
                "FS={:.4f}/{:.4f}/{:.4f}({}{})".format(
                    row["rmsd_is_to_dft00_raw_A"],
                    row["rmsd_is_to_dft00_aligned_A"],
                    row["max_atom_rmsd_is_to_dft00_aligned_A"],
                    row["max_atom_rmsd_is_to_dft00_element"],
                    row["max_atom_rmsd_is_to_dft00_index"],
                    row["rmsd_ts_to_dft01_raw_A"],
                    row["rmsd_ts_to_dft01_aligned_A"],
                    row["max_atom_rmsd_ts_to_dft01_aligned_A"],
                    row["max_atom_rmsd_ts_to_dft01_element"],
                    row["max_atom_rmsd_ts_to_dft01_index"],
                    row["rmsd_fs_to_dft02_raw_A"],
                    row["rmsd_fs_to_dft02_aligned_A"],
                    row["max_atom_rmsd_fs_to_dft02_aligned_A"],
                    row["max_atom_rmsd_fs_to_dft02_element"],
                    row["max_atom_rmsd_fs_to_dft02_index"],
                )
            )

    csv_path = out_dir / "np_relax_neb_summary.csv"
    md_path = out_dir / "np_relax_neb_summary.md"

    write_csv(csv_path, rows)
    write_md(md_path, rows, args)

    print(f"CSV: {csv_path}")
    print(f"Summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

