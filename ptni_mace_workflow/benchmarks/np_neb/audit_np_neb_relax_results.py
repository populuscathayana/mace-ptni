#!/usr/bin/env python
"""Audit NP relax+NEB benchmark results for failed/unphysical paths.

The relax+NEB benchmark can produce misleading barrier numbers if an inner NEB
image reconstructs into a different channel, emits an atom, develops atom
overlap, or simply does not converge. This script combines the summary CSV with
the saved neb_initial.extxyz and neb_final.extxyz files and writes a compact
diagnostic table.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def to_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def parse_float_list(text: str) -> list[float]:
    values = []
    for part in str(text or "").split(";"):
        number = to_float(part)
        if math.isfinite(number):
            values.append(number)
    return values


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def nearest_neighbor_metrics(atoms: Any) -> dict[str, Any]:
    import numpy as np

    pos = atoms.get_positions()
    center = pos.mean(axis=0)
    radii = np.linalg.norm(pos - center, axis=1)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)

    max_radius_atom = int(radii.argmax())
    max_nn_atom = int(nn.argmax())
    min_nn_atom = int(nn.argmin())
    return {
        "max_radius_A": float(radii[max_radius_atom]),
        "max_radius_atom": max_radius_atom,
        "max_radius_symbol": atoms[max_radius_atom].symbol,
        "max_nn_A": float(nn[max_nn_atom]),
        "max_nn_atom": max_nn_atom,
        "max_nn_symbol": atoms[max_nn_atom].symbol,
        "min_nn_A": float(nn[min_nn_atom]),
        "min_nn_atom": min_nn_atom,
        "min_nn_symbol": atoms[min_nn_atom].symbol,
    }


def geometry_metrics(group_dir: Path) -> dict[str, Any]:
    from ase.io import read
    import numpy as np

    initial_path = group_dir / "neb_initial.extxyz"
    final_path = group_dir / "neb_final.extxyz"
    if not initial_path.is_file() or not final_path.is_file():
        return {"geometry_status": "missing_neb_extxyz"}

    initial = read(initial_path.as_posix(), index=":")
    final = read(final_path.as_posix(), index=":")
    if len(initial) != len(final):
        return {"geometry_status": "image_count_mismatch"}

    best_disp = {
        "max_inner_displacement_A": 0.0,
        "max_disp_image": "",
        "max_disp_atom": "",
        "max_disp_symbol": "",
    }
    best_nn = {
        "final_max_nn_A": 0.0,
        "final_max_nn_image": "",
        "final_max_nn_atom": "",
        "final_max_nn_symbol": "",
    }
    best_radius = {
        "final_max_radius_A": 0.0,
        "final_max_radius_image": "",
        "final_max_radius_atom": "",
        "final_max_radius_symbol": "",
    }
    min_nn = {
        "final_min_nn_A": float("inf"),
        "final_min_nn_image": "",
        "final_min_nn_atom": "",
        "final_min_nn_symbol": "",
    }

    inner_indices = range(1, max(len(final) - 1, 1))
    for image_index, (atoms0, atoms1) in enumerate(zip(initial, final)):
        if len(atoms0) != len(atoms1):
            return {"geometry_status": f"atom_count_mismatch_image_{image_index}"}
        if image_index in inner_indices:
            disp = np.linalg.norm(atoms1.get_positions() - atoms0.get_positions(), axis=1)
            atom_index = int(disp.argmax())
            if float(disp[atom_index]) > float(best_disp["max_inner_displacement_A"]):
                best_disp = {
                    "max_inner_displacement_A": float(disp[atom_index]),
                    "max_disp_image": image_index,
                    "max_disp_atom": atom_index,
                    "max_disp_symbol": atoms1[atom_index].symbol,
                }

        metrics = nearest_neighbor_metrics(atoms1)
        if metrics["max_nn_A"] > float(best_nn["final_max_nn_A"]):
            best_nn = {
                "final_max_nn_A": metrics["max_nn_A"],
                "final_max_nn_image": image_index,
                "final_max_nn_atom": metrics["max_nn_atom"],
                "final_max_nn_symbol": metrics["max_nn_symbol"],
            }
        if metrics["max_radius_A"] > float(best_radius["final_max_radius_A"]):
            best_radius = {
                "final_max_radius_A": metrics["max_radius_A"],
                "final_max_radius_image": image_index,
                "final_max_radius_atom": metrics["max_radius_atom"],
                "final_max_radius_symbol": metrics["max_radius_symbol"],
            }
        if metrics["min_nn_A"] < float(min_nn["final_min_nn_A"]):
            min_nn = {
                "final_min_nn_A": metrics["min_nn_A"],
                "final_min_nn_image": image_index,
                "final_min_nn_atom": metrics["min_nn_atom"],
                "final_min_nn_symbol": metrics["min_nn_symbol"],
            }

    return {
        "geometry_status": "ok",
        **best_disp,
        **best_nn,
        **best_radius,
        **min_nn,
    }


def classify(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    flags = []
    notes = []

    path_force = to_float(row.get("neb_max_force_path_eVA"))
    if math.isfinite(path_force) and path_force > args.force_warn:
        flags.append("high_path_force")
        notes.append(f"path force {path_force:.3g} eV/A > {args.force_warn:g}")

    sp_err = abs(to_float(row.get("abs_error_sp_forward_barrier_eV")))
    neb_err = abs(to_float(row.get("abs_error_neb_forward_barrier_eV")))
    if math.isfinite(sp_err) and math.isfinite(neb_err) and (neb_err - sp_err) > args.neb_sp_gap_warn:
        flags.append("neb_sp_divergence")
        notes.append(f"NEB error exceeds SP error by {neb_err - sp_err:.3g} eV")

    max_nn = to_float(row.get("final_max_nn_A"))
    if math.isfinite(max_nn) and max_nn > args.detached_nn:
        flags.append("detached_atom")
        notes.append(f"max nearest-neighbor distance {max_nn:.3g} A")

    max_disp = to_float(row.get("max_inner_displacement_A"))
    if math.isfinite(max_disp) and max_disp > args.displacement_warn:
        flags.append("large_inner_displacement")
        notes.append(f"max inner-image displacement {max_disp:.3g} A")

    min_nn = to_float(row.get("final_min_nn_A"))
    if math.isfinite(min_nn) and min_nn < args.collision_nn:
        flags.append("atom_overlap")
        notes.append(f"minimum nearest-neighbor distance {min_nn:.3g} A")

    if not flags:
        flags.append("ok_or_review")
    return ";".join(flags), " | ".join(notes)


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


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if math.isfinite(number):
        return f"{number:.{digits}f}"
    return str(value or "")


def write_md(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        for flag in str(row.get("audit_flags", "")).split(";"):
            counts[flag] = counts.get(flag, 0) + 1

    by_neb_error = sorted(rows, key=lambda row: to_float(row.get("abs_error_neb_forward_barrier_eV")), reverse=True)
    by_detached = sorted(rows, key=lambda row: to_float(row.get("final_max_nn_A")), reverse=True)
    by_force = sorted(rows, key=lambda row: to_float(row.get("neb_max_force_path_eVA")), reverse=True)

    lines = [
        "# NP Relax+NEB Audit",
        "",
        f"- Input CSV: `{args.summary_csv.resolve()}`",
        f"- Rows audited: {len(rows)}",
        f"- Detached atom threshold: nearest-neighbor distance > {args.detached_nn:g} A",
        f"- Atom overlap threshold: nearest-neighbor distance < {args.collision_nn:g} A",
        f"- Large displacement threshold: inner image displacement > {args.displacement_warn:g} A",
        f"- Path force warning threshold: raw max image force > {args.force_warn:g} eV/A",
        "",
        "## Flag Counts",
        "",
        "| flag | count |",
        "|---|---:|",
    ]
    for flag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {flag} | {count} |")

    def table(title: str, subset: list[dict[str, Any]]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| rank | group | flags | SP err eV | NEB err eV | path force eV/A | max NN A | max disp A |",
                "|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(subset[:12], start=1):
            lines.append(
                f"| {rank} | `{row.get('neb_group','')}` | {row.get('audit_flags','')} | "
                f"{fmt(row.get('abs_error_sp_forward_barrier_eV'))} | "
                f"{fmt(row.get('abs_error_neb_forward_barrier_eV'))} | "
                f"{fmt(row.get('neb_max_force_path_eVA'))} | "
                f"{fmt(row.get('final_max_nn_A'))} | "
                f"{fmt(row.get('max_inner_displacement_A'))} |"
            )

    table("Largest NEB Barrier Errors", by_neb_error)
    table("Most Detached Atoms", by_detached)
    table("Largest Raw Path Forces", by_force)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit NP relax+NEB result CSV and saved geometries.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--force-warn", type=float, default=1.0)
    parser.add_argument("--detached-nn", type=float, default=4.0)
    parser.add_argument("--collision-nn", type=float, default=1.8)
    parser.add_argument("--displacement-warn", type=float, default=5.0)
    parser.add_argument("--neb-sp-gap-warn", type=float, default=0.5)
    args = parser.parse_args()

    rows = read_csv(args.summary_csv.resolve())
    audit_rows = []
    for index, row in enumerate(rows, start=1):
        group_dir = Path(str(row.get("group_dir", "")))
        if group_dir.as_posix().startswith("/mnt/c/"):
            # Path is already valid inside WSL. On Windows, try the relative path
            # stored after the project folder if needed.
            pass
        geom = geometry_metrics(group_dir) if group_dir.exists() else {"geometry_status": "missing_group_dir"}
        merged = {**row, **geom}
        merged["audit_index"] = index
        merged["audit_flags"], merged["audit_notes"] = classify(merged, args)
        energies = parse_float_list(row.get("neb_energies_relative_eV", ""))
        if energies:
            merged["neb_relative_energy_span_eV"] = max(energies) - min(energies)
        audit_rows.append(merged)

    write_csv(args.out_csv.resolve(), audit_rows)
    write_md(args.out_md.resolve(), audit_rows, args)

    problem_rows = [row for row in audit_rows if row.get("audit_flags") != "ok_or_review"]
    print(f"Rows audited: {len(audit_rows)}")
    print(f"Rows with warning flags: {len(problem_rows)}")
    print(f"CSV: {args.out_csv.resolve()}")
    print(f"Markdown: {args.out_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
