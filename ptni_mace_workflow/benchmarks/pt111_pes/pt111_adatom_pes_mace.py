#!/usr/bin/env python
"""Reproduce a Pt(111) adatom potential-energy surface with MACE.

For each hex-grid VASP directory, this script reads the POSCAR, fixes every
atom except the last one, constrains the last atom to move only along z, relaxes
that one degree of freedom with MACE, and writes a PES table/plots.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def point_index_from_path(path: Path) -> int | None:
    match = re.search(r"hex_point_(\d+)", path.parent.name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", path.parent.name)
    if match:
        return int(match.group(1))
    return None


def point_label_from_path(path: Path) -> str:
    index = point_index_from_path(path)
    if index is not None:
        return f"hex_point_{index:03d}"
    return path.parent.name


def point_sort_key(path: Path) -> tuple[int, int | str, str]:
    index = point_index_from_path(path)
    if index is not None:
        return (0, index, str(path))
    return (1, path.parent.name, str(path))


def find_poscars(input_dir: Path, pattern: str, max_points: int | None = None) -> list[Path]:
    paths = sorted((p for p in input_dir.glob(pattern) if p.is_file()), key=point_sort_key)
    if not paths:
        raise SystemExit(f"No POSCAR files found under {input_dir} with pattern {pattern!r}")
    if max_points is not None:
        paths = paths[:max_points]
    return paths


def read_energy_table(path: Path) -> dict[int, float]:
    energies: dict[int, float] = {}
    if not path.exists():
        return energies
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                index = int(parts[0])
                energy = float(parts[1])
            except ValueError:
                continue
            energies[index] = energy
    return energies


def read_oszicar_energy(path: Path) -> float | None:
    if not path.exists():
        return None
    pattern = re.compile(rf"\bE0=\s*({FLOAT_RE})")
    last_energy = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                last_energy = float(match.group(1))
    return last_energy


def read_outcar_energy(path: Path) -> float | None:
    if not path.exists():
        return None
    patterns = [
        re.compile(rf"free\s+energy\s+TOTEN\s*=\s*({FLOAT_RE})\s+eV"),
        re.compile(rf"energy\s+without\s+entropy\s*=\s*({FLOAT_RE})"),
    ]
    last_energy = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    last_energy = float(match.group(1))
    return last_energy


def dft_energy_for_poscar(poscar: Path, source: str, energy_table: dict[int, float]) -> tuple[float | None, str]:
    index = point_index_from_path(poscar)
    if source in {"auto", "energies_txt"} and index is not None and index in energy_table:
        return energy_table[index], "energies.txt"
    if source == "energies_txt":
        return None, "missing energies.txt entry"

    if source in {"auto", "oszicar"}:
        energy = read_oszicar_energy(poscar.parent / "OSZICAR")
        if energy is not None:
            return energy, "OSZICAR"
        if source == "oszicar":
            return None, "missing OSZICAR E0"

    if source in {"auto", "outcar"}:
        energy = read_outcar_energy(poscar.parent / "OUTCAR")
        if energy is not None:
            return energy, "OUTCAR"
        if source == "outcar":
            return None, "missing OUTCAR energy"

    return None, "none"


def max_norm(vectors) -> float:
    import numpy as np

    if len(vectors) == 0:
        return 0.0
    return float(np.linalg.norm(vectors, axis=1).max())


def get_adatom_state(atoms) -> dict[str, float | str]:
    import numpy as np

    index = len(atoms) - 1
    pos = atoms.get_positions()[index]
    scaled = atoms.get_scaled_positions(wrap=False)[index]
    scaled_wrapped = scaled.copy()
    for axis in (0, 1):
        if atoms.pbc[axis]:
            scaled_wrapped[axis] = scaled_wrapped[axis] % 1.0
    return {
        "adatom_index": index,
        "adatom_element": atoms.get_chemical_symbols()[index],
        "cart_x_A": float(pos[0]),
        "cart_y_A": float(pos[1]),
        "cart_z_A": float(pos[2]),
        "frac_a": float(scaled[0]),
        "frac_b": float(scaled[1]),
        "frac_c": float(scaled[2]),
        "frac_a_wrapped": float(scaled_wrapped[0]),
        "frac_b_wrapped": float(scaled_wrapped[1]),
        "cell_a_A": float(np.linalg.norm(atoms.cell.array[0])),
        "cell_b_A": float(np.linalg.norm(atoms.cell.array[1])),
        "cell_c_A": float(np.linalg.norm(atoms.cell.array[2])),
    }


def apply_adatom_z_constraint(atoms):
    from ase.constraints import FixAtoms, FixedLine

    last_index = len(atoms) - 1
    constraints = []
    if last_index > 0:
        constraints.append(FixAtoms(indices=list(range(last_index))))
    constraints.append(FixedLine(last_index, [0.0, 0.0, 1.0]))
    atoms.set_constraint(constraints)


def make_optimizer(name: str, atoms, trajectory: Path | None, logfile: Path | None):
    from ase.optimize import BFGS, FIRE, LBFGS

    classes = {"BFGS": BFGS, "FIRE": FIRE, "LBFGS": LBFGS}
    try:
        opt_cls = classes[name.upper()]
    except KeyError as exc:
        raise SystemExit(f"Unknown optimizer {name!r}. Choose one of: {', '.join(classes)}") from exc
    return opt_cls(
        atoms,
        trajectory=str(trajectory) if trajectory else None,
        logfile=str(logfile) if logfile else None,
    )


def set_adatom_z(atoms, z_value: float) -> None:
    positions = atoms.get_positions()
    positions[-1, 2] = z_value
    atoms.set_positions(positions, apply_constraint=False)


def minimize_adatom_z_by_energy(
    atoms,
    z0: float,
    z_window: float,
    energy_tol: float,
    z_tol: float,
    max_evals: int,
    min_evals: int,
) -> dict[str, float | int | str | bool]:
    """Bounded golden-section minimization along the adatom z coordinate."""
    if z_window <= 0:
        raise ValueError("z_window must be positive")
    if max_evals < 8:
        raise ValueError("max_evals must be >= 8 for energy-mode relaxation")

    gr = (math.sqrt(5.0) - 1.0) / 2.0
    left = z0 - z_window
    right = z0 + z_window
    c = right - gr * (right - left)
    d = left + gr * (right - left)

    def energy_at(z_value: float) -> float:
        set_adatom_z(atoms, z_value)
        return float(atoms.get_potential_energy())

    ec = energy_at(c)
    ed = energy_at(d)
    evals = 2
    best_z = c if ec <= ed else d
    best_energy = min(ec, ed)
    prev_best_energy = best_energy
    stall_count = 0
    reason = "max_evals"

    while evals < max_evals:
        if abs(right - left) <= z_tol and evals >= min_evals:
            reason = "z_tol"
            break

        if ec <= ed:
            right = d
            d = c
            ed = ec
            c = right - gr * (right - left)
            ec = energy_at(c)
            candidate_z = c
            candidate_energy = ec
        else:
            left = c
            c = d
            ec = ed
            d = left + gr * (right - left)
            ed = energy_at(d)
            candidate_z = d
            candidate_energy = ed
        evals += 1

        if candidate_energy < best_energy:
            best_z = candidate_z
            best_energy = candidate_energy

        if abs(prev_best_energy - best_energy) < energy_tol:
            stall_count += 1
        else:
            stall_count = 0
        prev_best_energy = best_energy

        if evals >= min_evals and stall_count >= 3:
            reason = "energy_tol"
            break

    set_adatom_z(atoms, best_z)
    # Ensure the final reported energy is evaluated at the selected best z.
    final_energy = float(atoms.get_potential_energy())
    return {
        "best_z_A": best_z,
        "energy_eV": final_energy,
        "energy_evaluations": evals,
        "converged": reason != "max_evals",
        "convergence_reason": reason,
        "final_bracket_width_A": abs(right - left),
        "energy_tol_eV": energy_tol,
        "z_tol_A": z_tol,
        "z_window_A": z_window,
    }


def relax_one_poscar(
    poscar: Path,
    calc,
    fmax: float,
    steps: int,
    optimizer_name: str,
    save_trajectory: bool,
    point_dir: Path,
    relax_mode: str,
    energy_tol: float,
    z_tol: float,
    z_window: float,
    max_evals: int,
    min_evals: int,
):
    from ase.io import read

    atoms = read(str(poscar), format="vasp")
    atoms.set_pbc(True)
    initial_state = get_adatom_state(atoms)
    apply_adatom_z_constraint(atoms)
    atoms.calc = calc

    initial_energy = float(atoms.get_potential_energy())
    initial_forces_raw = atoms.get_forces(apply_constraint=False)
    initial_forces_constrained = atoms.get_forces(apply_constraint=True)
    initial_adatom_fz = float(initial_forces_raw[len(atoms) - 1][2])
    initial_max_force = max_norm(initial_forces_constrained)

    converged = None
    nsteps = 0
    convergence_reason = ""
    energy_result: dict[str, float | int | str | bool] = {}
    if relax_mode == "force" and steps > 0:
        trajectory = point_dir / "relax.traj" if save_trajectory else None
        logfile = point_dir / "relax.log"
        opt = make_optimizer(optimizer_name, atoms, trajectory=trajectory, logfile=logfile)
        converged = bool(opt.run(fmax=fmax, steps=steps))
        nsteps = int(opt.get_number_of_steps())
        convergence_reason = "fmax" if converged else "max_steps"
    elif relax_mode == "energy":
        energy_result = minimize_adatom_z_by_energy(
            atoms=atoms,
            z0=float(initial_state["cart_z_A"]),
            z_window=z_window,
            energy_tol=energy_tol,
            z_tol=z_tol,
            max_evals=max_evals,
            min_evals=min_evals,
        )
        converged = bool(energy_result["converged"])
        nsteps = int(energy_result["energy_evaluations"])
        convergence_reason = str(energy_result["convergence_reason"])
    elif relax_mode != "force":
        raise SystemExit(f"Unknown relax mode {relax_mode!r}; choose force or energy")

    final_energy = float(atoms.get_potential_energy())
    final_forces_raw = atoms.get_forces(apply_constraint=False)
    final_forces_constrained = atoms.get_forces(apply_constraint=True)
    final_state = get_adatom_state(atoms)
    final_adatom_fz = float(final_forces_raw[len(atoms) - 1][2])
    final_max_force = max_norm(final_forces_constrained)
    atoms.calc = None

    return {
        "atoms": atoms,
        "initial_state": initial_state,
        "final_state": final_state,
        "initial_energy_eV": initial_energy,
        "final_energy_eV": final_energy,
        "delta_relax_energy_eV": final_energy - initial_energy,
        "initial_max_force_constrained_eVA": initial_max_force,
        "final_max_force_constrained_eVA": final_max_force,
        "initial_adatom_force_z_raw_eVA": initial_adatom_fz,
        "final_adatom_force_z_raw_eVA": final_adatom_fz,
        "converged": converged,
        "convergence_reason": convergence_reason,
        "relax_mode": relax_mode,
        "optimizer_steps": nsteps,
        "energy_evaluations": energy_result.get("energy_evaluations", ""),
        "final_bracket_width_A": energy_result.get("final_bracket_width_A", ""),
        "energy_tol_eV": energy_result.get("energy_tol_eV", ""),
        "z_tol_A": energy_result.get("z_tol_A", ""),
        "z_window_A": energy_result.get("z_window_A", ""),
    }


def read_dft_final_z(poscar: Path) -> float | None:
    try:
        from ase.io import read
    except Exception:
        return None
    contcar = poscar.parent / "CONTCAR"
    if not contcar.exists():
        return None
    try:
        atoms = read(str(contcar), format="vasp")
    except Exception:
        return None
    return float(atoms.get_positions()[-1][2])


def write_relaxed_structures(out_dir: Path, label: str, atoms) -> tuple[str, str]:
    from ase.io import write

    point_dir = out_dir / label
    point_dir.mkdir(parents=True, exist_ok=True)
    poscar_path = point_dir / "POSCAR_mace_relaxed"
    extxyz_path = point_dir / "mace_relaxed.extxyz"
    write(str(poscar_path), atoms, format="vasp", direct=True, vasp5=True, sort=False)
    write(str(extxyz_path), atoms, format="extxyz")
    return str(poscar_path), str(extxyz_path)


def load_existing_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sort_value_or_inf(value) -> float:
    parsed = to_float(value)
    return parsed if parsed is not None else math.inf


def add_relative_energies(rows: list[dict]) -> None:
    mace_values = [to_float(row.get("mace_final_energy_eV")) for row in rows]
    mace_values = [value for value in mace_values if value is not None]
    dft_values = [to_float(row.get("dft_energy_eV")) for row in rows]
    dft_values = [value for value in dft_values if value is not None]
    mace_min = min(mace_values) if mace_values else None
    dft_min = min(dft_values) if dft_values else None

    for row in rows:
        mace_energy = to_float(row.get("mace_final_energy_eV"))
        dft_energy = to_float(row.get("dft_energy_eV"))
        if mace_energy is not None and mace_min is not None:
            rel = mace_energy - mace_min
            row["mace_rel_eV"] = rel
            row["mace_rel_meV"] = rel * 1000.0
        if dft_energy is not None and dft_min is not None:
            rel = dft_energy - dft_min
            row["dft_rel_eV"] = rel
            row["dft_rel_meV"] = rel * 1000.0
        mace_rel = to_float(row.get("mace_rel_eV"))
        dft_rel = to_float(row.get("dft_rel_eV"))
        if mace_rel is not None and dft_rel is not None:
            diff = mace_rel - dft_rel
            row["mace_minus_dft_rel_eV"] = diff
            row["mace_minus_dft_rel_meV"] = diff * 1000.0


def metric_summary(rows: list[dict]) -> dict[str, float | str | int | None]:
    import numpy as np

    diffs = [to_float(row.get("mace_minus_dft_rel_meV")) for row in rows]
    diffs = [value for value in diffs if value is not None]
    mace_rel = [to_float(row.get("mace_rel_meV")) for row in rows]
    mace_rel = [value for value in mace_rel if value is not None]
    dft_rel = [to_float(row.get("dft_rel_meV")) for row in rows]
    dft_rel = [value for value in dft_rel if value is not None]

    summary: dict[str, float | str | int | None] = {
        "n_points": len(rows),
        "n_points_with_dft": len(diffs),
        "mace_pes_range_meV": max(mace_rel) - min(mace_rel) if mace_rel else None,
        "dft_pes_range_meV": max(dft_rel) - min(dft_rel) if dft_rel else None,
    }
    if diffs:
        arr = np.asarray(diffs, dtype=float)
        summary.update(
            {
                "rel_energy_mae_meV": float(np.mean(np.abs(arr))),
                "rel_energy_rmse_meV": float(np.sqrt(np.mean(arr * arr))),
                "rel_energy_max_abs_meV": float(np.max(np.abs(arr))),
            }
        )
    if len(mace_rel) == len(dft_rel) and len(mace_rel) >= 2:
        corr = float(np.corrcoef(np.asarray(dft_rel), np.asarray(mace_rel))[0, 1])
        summary["pearson_r_rel_energy"] = corr

    if rows:
        mace_min_row = min(rows, key=lambda row: sort_value_or_inf(row.get("mace_rel_meV")))
        summary["mace_min_label"] = mace_min_row.get("label")
        dft_rows = [row for row in rows if to_float(row.get("dft_rel_meV")) is not None]
        if dft_rows:
            dft_min_row = min(dft_rows, key=lambda row: sort_value_or_inf(row.get("dft_rel_meV")))
            summary["dft_min_label"] = dft_min_row.get("label")
    return summary


def plot_surface(
    rows: list[dict],
    value_key: str,
    path: Path,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    center_zero: bool = False,
) -> bool:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        import numpy as np
    except Exception:
        return False

    points = []
    values = []
    labels = []
    for row in rows:
        x = to_float(row.get("initial_cart_x_A"))
        y = to_float(row.get("initial_cart_y_A"))
        value = to_float(row.get(value_key))
        if x is None or y is None or value is None:
            continue
        points.append((x, y))
        values.append(value)
        labels.append(row.get("label", ""))
    if len(points) < 3:
        return False

    xy = np.asarray(points, dtype=float)
    z = np.asarray(values, dtype=float)
    if center_zero:
        bound = float(np.max(np.abs(z))) if len(z) else 1.0
        if bound == 0.0:
            bound = 1.0
        vmin, vmax = -bound, bound

    fig, ax = plt.subplots(figsize=(7, 6))
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
    try:
        contour = ax.tricontourf(triang, z, levels=24, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.tricontour(triang, z, levels=12, colors="k", linewidths=0.25, alpha=0.35)
    except Exception:
        contour = ax.scatter(xy[:, 0], xy[:, 1], c=z, cmap=cmap, vmin=vmin, vmax=vmax, s=38)
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=z, cmap=cmap, vmin=vmin, vmax=vmax, s=18, edgecolor="k", linewidth=0.2)
    del scatter
    min_i = int(np.argmin(z))
    max_i = int(np.argmax(z))
    ax.scatter([xy[min_i, 0]], [xy[min_i, 1]], marker="*", s=120, color="white", edgecolor="black", label=f"min {labels[min_i]}")
    ax.scatter([xy[max_i, 0]], [xy[max_i, 1]], marker="X", s=80, color="tab:red", edgecolor="black", label=f"max {labels[max_i]}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("adatom x (A)")
    ax.set_ylabel("adatom y (A)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("relative energy (meV)" if "rel" in value_key else value_key)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def write_markdown(
    path: Path,
    args,
    rows: list[dict],
    summary: dict[str, float | str | int | None],
    csv_path: Path,
    plot_paths: Iterable[Path],
) -> None:
    lines = [
        "# Pt(111) Adatom PES With MACE",
        "",
        f"- Input directory: `{args.input_dir.resolve()}`",
        f"- POSCAR pattern: `{args.pattern}`",
        f"- Model: `{args.model.resolve()}`",
        f"- Device: `{args.device}`",
        f"- dtype: `{args.default_dtype}`",
        f"- Constraint: all atoms fixed except the last atom; last atom fixed in x/y and relaxed only along z.",
        f"- Optimizer: `{args.optimizer}`, fmax={args.fmax}, steps={args.steps}",
        f"- DFT energy source: `{args.dft_energy_source}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Lowest-Energy Points",
            "",
            "| rank | MACE label | MACE rel meV | DFT label | DFT rel meV |",
            "|---:|---|---:|---|---:|",
        ]
    )
    mace_sorted = sorted(rows, key=lambda row: sort_value_or_inf(row.get("mace_rel_meV")))
    dft_sorted = sorted(
        [row for row in rows if to_float(row.get("dft_rel_meV")) is not None],
        key=lambda row: sort_value_or_inf(row.get("dft_rel_meV")),
    )
    for i in range(min(10, len(mace_sorted))):
        mace_row = mace_sorted[i]
        dft_row = dft_sorted[i] if i < len(dft_sorted) else {}
        lines.append(
            f"| {i + 1} | {mace_row.get('label', '')} | {to_float(mace_row.get('mace_rel_meV')):.3f} | "
            f"{dft_row.get('label', '')} | "
            f"{to_float(dft_row.get('dft_rel_meV')):.3f} |"
            if dft_row
            else f"| {i + 1} | {mace_row.get('label', '')} | {to_float(mace_row.get('mace_rel_meV')):.3f} |  |  |"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- CSV: `{csv_path}`",
        ]
    )
    for plot_path in plot_paths:
        lines.append(f"- Plot: `{plot_path}`")
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- This is a constrained z-relaxation PES, not a full unconstrained adatom diffusion calculation.",
            "- Compare relative PES shapes and site ordering before interpreting absolute total energies.",
            "- If MACE and DFT minima differ, inspect the corresponding relaxed z positions and local forces first.",
            "- This test is a targeted validation set and should stay outside training unless deliberately used for active learning.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="MACE Pt(111) adatom hex-grid PES with z-only relaxation.")
    parser.add_argument("--input-dir", type=Path, default=Path("mace_workspace/inputs/pt111"))
    parser.add_argument("--pattern", default="**/POSCAR", help="POSCAR glob relative to input dir. Default: **/POSCAR.")
    parser.add_argument("--model", type=Path, required=True, help="MACE .model file, not compiled.model.")
    parser.add_argument("--out-dir", type=Path, default=Path("mace_workspace/runs/benchmarks/pt111_pes/manual"))
    parser.add_argument("--name", default="pt111_adatom_pes")
    parser.add_argument(
        "--dft-energy-source",
        choices=["auto", "energies_txt", "oszicar", "outcar", "none"],
        default="auto",
        help="DFT reference energy source. auto uses energies.txt, then OSZICAR, then OUTCAR.",
    )
    parser.add_argument("--energies-txt", type=Path, default=None, help="Optional energy table. Default: input-dir/energies.txt.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", default="float64")
    parser.add_argument("--optimizer", choices=["FIRE", "BFGS", "LBFGS"], default="FIRE")
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--relax-mode",
        choices=["force", "energy"],
        default="force",
        help="force uses ASE optimizer/fmax; energy uses 1D bounded z minimization and energy_tol.",
    )
    parser.add_argument("--energy-tol", type=float, default=1.0e-5, help="Energy-mode convergence tolerance in eV.")
    parser.add_argument("--z-tol", type=float, default=1.0e-5, help="Energy-mode z bracket tolerance in A.")
    parser.add_argument("--z-window", type=float, default=1.0, help="Energy-mode z search half-window in A around initial z.")
    parser.add_argument("--max-evals", type=int, default=80, help="Energy-mode maximum MACE energy evaluations per point.")
    parser.add_argument("--min-evals", type=int, default=10, help="Energy-mode minimum evaluations before convergence checks.")
    parser.add_argument("--max-points", type=int, default=None, help="Only process the first N sorted points.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split sorted POSCAR list into this many shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Run only this zero-based shard index.")
    parser.add_argument("--write-relaxed", action="store_true", help="Write relaxed POSCAR/extxyz for each grid point.")
    parser.add_argument("--save-trajectory", action="store_true", help="Write ASE trajectory for each z relaxation.")
    parser.add_argument("--plot", action="store_true", help="Write PNG PES plots with matplotlib.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing CSV rows and skip completed labels.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing result CSV.")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.name}_results.csv"
    md_path = out_dir / f"{args.name}_summary.md"

    if csv_path.exists() and not args.resume and not args.overwrite:
        raise SystemExit(f"Result CSV already exists: {csv_path}. Use --resume or --overwrite.")
    rows: list[dict] = [] if args.overwrite else load_existing_rows(csv_path)
    completed = {row.get("label") for row in rows if row.get("label")} if args.resume else set()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")

    poscars_all = find_poscars(input_dir, args.pattern, args.max_points)
    poscars = [path for idx, path in enumerate(poscars_all) if idx % args.num_shards == args.shard_index]
    if not poscars:
        raise SystemExit(f"Shard {args.shard_index}/{args.num_shards} has no POSCAR files")
    energies_txt = args.energies_txt.resolve() if args.energies_txt else input_dir / "energies.txt"
    energy_table = read_energy_table(energies_txt) if args.dft_energy_source in {"auto", "energies_txt"} else {}

    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=str(args.model.resolve()),
        device=args.device,
        default_dtype=args.default_dtype,
    )

    print(f"Input dir: {input_dir}")
    print(f"POSCAR files: {len(poscars)} in shard {args.shard_index}/{args.num_shards} (total before shard: {len(poscars_all)})")
    print(f"Existing completed rows: {len(completed)}")
    print(f"Output CSV: {csv_path}")

    for i, poscar in enumerate(poscars, start=1):
        label = point_label_from_path(poscar)
        if label in completed:
            print(f"[{i}/{len(poscars)}] skip completed {label}")
            continue

        point_out_dir = out_dir / label
        point_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(poscars)}] relaxing {label}: {poscar}")

        dft_energy, dft_source = dft_energy_for_poscar(poscar, args.dft_energy_source, energy_table)
        result = relax_one_poscar(
            poscar=poscar,
            calc=calc,
            fmax=args.fmax,
            steps=args.steps,
            optimizer_name=args.optimizer,
            save_trajectory=args.save_trajectory,
            point_dir=point_out_dir,
            relax_mode=args.relax_mode,
            energy_tol=args.energy_tol,
            z_tol=args.z_tol,
            z_window=args.z_window,
            max_evals=args.max_evals,
            min_evals=args.min_evals,
        )

        atoms = result["atoms"]
        initial_state = result["initial_state"]
        final_state = result["final_state"]
        dft_final_z = read_dft_final_z(poscar)
        relaxed_poscar = ""
        relaxed_extxyz = ""
        if args.write_relaxed:
            relaxed_poscar, relaxed_extxyz = write_relaxed_structures(out_dir, label, atoms)

        row = {
            "label": label,
            "point_index": point_index_from_path(poscar),
            "poscar": str(poscar.resolve()),
            "natoms": len(atoms),
            "adatom_index": initial_state["adatom_index"],
            "adatom_element": initial_state["adatom_element"],
            "initial_cart_x_A": initial_state["cart_x_A"],
            "initial_cart_y_A": initial_state["cart_y_A"],
            "initial_cart_z_A": initial_state["cart_z_A"],
            "final_cart_x_A": final_state["cart_x_A"],
            "final_cart_y_A": final_state["cart_y_A"],
            "final_cart_z_A": final_state["cart_z_A"],
            "delta_z_A": float(final_state["cart_z_A"]) - float(initial_state["cart_z_A"]),
            "dft_final_cart_z_A": dft_final_z,
            "mace_minus_dft_final_z_A": float(final_state["cart_z_A"]) - dft_final_z if dft_final_z is not None else None,
            "frac_a": initial_state["frac_a"],
            "frac_b": initial_state["frac_b"],
            "frac_c_initial": initial_state["frac_c"],
            "frac_c_final": final_state["frac_c"],
            "frac_a_wrapped": initial_state["frac_a_wrapped"],
            "frac_b_wrapped": initial_state["frac_b_wrapped"],
            "cell_a_A": initial_state["cell_a_A"],
            "cell_b_A": initial_state["cell_b_A"],
            "cell_c_A": initial_state["cell_c_A"],
            "mace_initial_energy_eV": result["initial_energy_eV"],
            "mace_final_energy_eV": result["final_energy_eV"],
            "mace_delta_relax_energy_eV": result["delta_relax_energy_eV"],
            "dft_energy_eV": dft_energy,
            "dft_energy_source": dft_source,
            "initial_max_force_constrained_eVA": result["initial_max_force_constrained_eVA"],
            "final_max_force_constrained_eVA": result["final_max_force_constrained_eVA"],
            "initial_adatom_force_z_raw_eVA": result["initial_adatom_force_z_raw_eVA"],
            "final_adatom_force_z_raw_eVA": result["final_adatom_force_z_raw_eVA"],
            "converged": result["converged"],
            "convergence_reason": result["convergence_reason"],
            "relax_mode": result["relax_mode"],
            "optimizer": args.optimizer,
            "optimizer_steps": result["optimizer_steps"],
            "energy_evaluations": result["energy_evaluations"],
            "final_bracket_width_A": result["final_bracket_width_A"],
            "energy_tol_eV": result["energy_tol_eV"],
            "z_tol_A": result["z_tol_A"],
            "z_window_A": result["z_window_A"],
            "relaxed_poscar": relaxed_poscar,
            "relaxed_extxyz": relaxed_extxyz,
        }
        rows.append(row)
        add_relative_energies(rows)
        write_rows(csv_path, rows)

        mace_rel = to_float(row.get("mace_rel_meV"))
        dft_rel = to_float(row.get("dft_rel_meV"))
        print(
            f"    E_MACE={row['mace_final_energy_eV']:.8f} eV; "
            f"rel={mace_rel:.3f} meV; "
            f"DFT_rel={dft_rel:.3f} meV" if dft_rel is not None else
            f"    E_MACE={row['mace_final_energy_eV']:.8f} eV; rel={mace_rel:.3f} meV"
        )

    add_relative_energies(rows)
    write_rows(csv_path, rows)
    summary = metric_summary(rows)

    plot_paths: list[Path] = []
    if args.plot:
        mace_values = [to_float(row.get("mace_rel_meV")) for row in rows]
        dft_values = [to_float(row.get("dft_rel_meV")) for row in rows]
        all_rel = [value for value in mace_values + dft_values if value is not None]
        shared_vmax = max(all_rel) if all_rel else None
        shared_vmin = 0.0 if all_rel else None

        mace_png = out_dir / f"{args.name}_mace_pes.png"
        if plot_surface(rows, "mace_rel_meV", mace_png, "MACE Pt(111) adatom PES", shared_vmin, shared_vmax):
            plot_paths.append(mace_png)
        dft_png = out_dir / f"{args.name}_dft_pes.png"
        if plot_surface(rows, "dft_rel_meV", dft_png, "DFT Pt(111) adatom PES", shared_vmin, shared_vmax):
            plot_paths.append(dft_png)
        diff_png = out_dir / f"{args.name}_mace_minus_dft_pes.png"
        if plot_surface(
            rows,
            "mace_minus_dft_rel_meV",
            diff_png,
            "MACE minus DFT relative PES",
            cmap="coolwarm",
            center_zero=True,
        ):
            plot_paths.append(diff_png)

    write_markdown(md_path, args, rows, summary, csv_path, plot_paths)
    print(f"\nCSV: {csv_path}")
    print(f"Summary: {md_path}")
    for plot_path in plot_paths:
        print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
