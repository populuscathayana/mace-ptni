#!/usr/bin/env python
"""Scan the vertical separation of the single movable atom in a PtNi slab.

The input POSCAR is expected to use VASP selective dynamics with one atom marked
as movable, typically `F F T`. The script extends vacuum along the third lattice
vector without stretching the slab, moves the selected atom along that same
direction, evaluates MACE single-point energies, and writes CSV/Markdown/plots.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


def parse_scan(text: str) -> list[float]:
    """Parse start:stop:step or comma-separated distances in Angstrom."""
    if ":" in text:
        parts = [float(part) for part in text.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("scan must be start:stop:step")
        start, stop, step = parts
        if step <= 0:
            raise argparse.ArgumentTypeError("scan step must be positive")
        values = []
        current = start
        while current <= stop + step * 1e-9:
            values.append(round(current, 10))
            current += step
        return values
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def is_int_line(text: str) -> bool:
    parts = text.split()
    if not parts:
        return False
    try:
        [int(part) for part in parts]
    except ValueError:
        return False
    return True


def selective_movable_indices(poscar: Path) -> list[int]:
    """Return atoms with any selective-dynamics T flag from a POSCAR."""
    lines = poscar.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 9:
        return []

    counts_line_index = 5 if is_int_line(lines[5]) else 6
    try:
        counts = [int(part) for part in lines[counts_line_index].split()]
    except ValueError:
        return []

    cursor = counts_line_index + 1
    has_selective = lines[cursor].strip().lower().startswith("s")
    if has_selective:
        cursor += 1
    if not has_selective:
        return []

    # Coordinate mode line.
    cursor += 1
    natoms = sum(counts)
    movable: list[int] = []
    for atom_index in range(natoms):
        if cursor + atom_index >= len(lines):
            break
        parts = lines[cursor + atom_index].split()
        if len(parts) < 6:
            continue
        flags = [part.upper().startswith("T") for part in parts[3:6]]
        if any(flags):
            movable.append(atom_index)
    return movable


def add_vacuum_along_c(atoms, vacuum_a: float):
    import numpy as np

    atoms = atoms.copy()
    cell = atoms.cell.array.copy()
    c_vec = cell[2]
    c_len = float(np.linalg.norm(c_vec))
    if c_len <= 0:
        raise ValueError("third lattice vector has zero length")
    unit_c = c_vec / c_len
    cell[2] = c_vec + vacuum_a * unit_c
    atoms.set_cell(cell, scale_atoms=False)
    return atoms


def cell_c_unit(atoms):
    import numpy as np

    c_vec = atoms.cell.array[2]
    c_len = float(np.linalg.norm(c_vec))
    if c_len <= 0:
        raise ValueError("third lattice vector has zero length")
    return c_vec / c_len, c_len


def infer_mobile_atom(atoms, poscar: Path, explicit_index: int | None) -> int:
    import numpy as np

    if explicit_index is not None:
        if explicit_index < 0 or explicit_index >= len(atoms):
            raise ValueError(f"--mobile-index out of range: {explicit_index}")
        return explicit_index

    movable = selective_movable_indices(poscar)
    if len(movable) == 1:
        return movable[0]
    if len(movable) > 1:
        raise ValueError(f"expected one movable atom from selective dynamics, found {movable}")

    unit_c, _ = cell_c_unit(atoms)
    projected = atoms.get_positions() @ unit_c
    return int(np.argmax(projected))


def nearest_lower_layer_spacing(atoms, mobile_index: int, layer_tol: float) -> float:
    import numpy as np

    unit_c, _ = cell_c_unit(atoms)
    projected = atoms.get_positions() @ unit_c
    mobile_value = float(projected[mobile_index])
    diffs = [
        mobile_value - float(value)
        for idx, value in enumerate(projected)
        if idx != mobile_index and mobile_value - float(value) > layer_tol
    ]
    if not diffs:
        raise ValueError("could not infer lower layer spacing along the third lattice vector")
    return float(min(diffs))


def shifted_structure(base_atoms, mobile_index: int, delta_a: float):
    atoms = base_atoms.copy()
    unit_c, _ = cell_c_unit(atoms)
    positions = atoms.get_positions()
    positions[mobile_index] = positions[mobile_index] + delta_a * unit_c
    atoms.set_positions(positions)
    return atoms


def maybe_plot_energy(
    path: Path,
    rows: list[dict],
    x_key: str,
    x_label: str,
    title: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x_values = [float(row[x_key]) for row in rows]
    y_values = [float(row["rel_energy_meV"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x_values, y_values, marker="o", markersize=3, linewidth=1.4)
    ax.set_xlabel(x_label)
    ax.set_ylabel("relative single-point energy (meV)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def maybe_plot_force(path: Path, rows: list[dict], title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x_values = [float(row["delta_A"]) for row in rows]
    axis_force = [float(row["mobile_force_axis_eVA"]) for row in rows]
    force_norm = [float(row["mobile_force_norm_eVA"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x_values, axis_force, marker="o", markersize=3, linewidth=1.4, label="axis component")
    ax.plot(x_values, force_norm, marker="s", markersize=3, linewidth=1.2, label="force norm")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("absolute displacement along cell c (A)")
    ax.set_ylabel("force on moved atom (eV/A)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def write_html_report(
    path: Path,
    title: str,
    summary_lines: list[str],
    rows: list[dict],
    image_paths: list[Path],
) -> None:
    rel_images = [image.name for image in image_paths if image.exists()]
    table_rows = "\n".join(
        "<tr>"
        f"<td>{row['delta_A']:.4f}</td>"
        f"<td>{row['delta_over_layer_spacing']:.4f}</td>"
        f"<td>{row['rel_energy_meV']:.6f}</td>"
        f"<td>{row['mobile_force_axis_eVA']:.6f}</td>"
        f"<td>{row['mobile_force_norm_eVA']:.6f}</td>"
        "</tr>"
        for row in rows
    )
    images_html = "\n".join(
        f'<figure><img src="{html.escape(image)}" alt="{html.escape(image)}"><figcaption>{html.escape(image)}</figcaption></figure>'
        for image in rel_images
    )
    summary_html = "\n".join(f"<li>{html.escape(line)}</li>" for line in summary_lines)
    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ margin-bottom: 0.25rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }}
    figure {{ margin: 0; }}
    img {{ max-width: 100%; border: 1px solid #d7dde5; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 6px 8px; text-align: right; }}
    th {{ background: #f3f5f8; }}
    td:first-child, th:first-child {{ text-align: right; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <ul>{summary_html}</ul>
  <div class="grid">{images_html}</div>
  <table>
    <thead>
      <tr>
        <th>delta A</th>
        <th>delta / layer spacing</th>
        <th>relative energy meV</th>
        <th>axis force eV/A</th>
        <th>force norm eV/A</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan a slab mobile atom distance using MACE single points.")
    parser.add_argument("--poscar", type=Path, required=True)
    parser.add_argument("--surface", default=None, help="Surface label, e.g. 111 or 100.")
    parser.add_argument("--model", type=Path, required=True, help="MACE .model file, not compiled.model.")
    parser.add_argument("--scan", type=parse_scan, default=parse_scan("0:5:0.05"), help="Distances in A. Default: 0:5:0.05.")
    parser.add_argument("--vacuum", type=float, default=20.0, help="Extra vacuum added along cell c in A. Default: 20.")
    parser.add_argument("--mobile-index", type=int, default=None, help="0-based atom index. Default: detect selective-dynamics movable atom.")
    parser.add_argument("--layer-tol", type=float, default=0.25, help="Minimum layer separation in A when inferring layer spacing.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", default="float64")
    parser.add_argument("--out-dir", type=Path, default=Path("mace_workspace/runs/benchmarks/distance_scan/manual"))
    parser.add_argument("--name", default="ptni_slab_distance_scan")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--write-structures", action="store_true")
    args = parser.parse_args()

    import numpy as np
    from ase.io import read, write
    from mace.calculators import MACECalculator

    if not args.poscar.is_file():
        raise SystemExit(f"POSCAR not found: {args.poscar}")
    if args.vacuum < 0:
        raise SystemExit("--vacuum must be >= 0")

    atoms0 = read(args.poscar.as_posix(), format="vasp")
    mobile_index = infer_mobile_atom(atoms0, args.poscar, args.mobile_index)
    base_atoms = add_vacuum_along_c(atoms0, args.vacuum)
    unit_c, c_len_after_vacuum = cell_c_unit(base_atoms)
    layer_spacing = nearest_lower_layer_spacing(base_atoms, mobile_index, args.layer_tol)
    base_scaled = base_atoms.get_scaled_positions(wrap=False)[mobile_index]
    mobile_element = base_atoms[mobile_index].symbol

    calc = MACECalculator(
        model_paths=str(args.model.resolve()),
        device=args.device,
        default_dtype=args.default_dtype,
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    structures_dir = out_dir / "structures"
    if args.write_structures:
        structures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    e0 = None
    for delta_a in args.scan:
        atoms = shifted_structure(base_atoms, mobile_index, delta_a)
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        mobile_force = forces[mobile_index]
        scaled = atoms.get_scaled_positions(wrap=False)[mobile_index]
        if e0 is None:
            e0 = energy

        row = {
            "surface": args.surface or args.poscar.parent.name,
            "model": str(args.model.resolve()),
            "poscar": str(args.poscar.resolve()),
            "natoms": len(atoms),
            "mobile_index_0based": mobile_index,
            "mobile_index_1based": mobile_index + 1,
            "mobile_element": mobile_element,
            "base_direct_z": float(base_scaled[2]),
            "direct_z": float(scaled[2]),
            "delta_A": float(delta_a),
            "delta_over_layer_spacing": float(delta_a / layer_spacing),
            "layer_spacing_A": float(layer_spacing),
            "cell_c_length_after_vacuum_A": float(c_len_after_vacuum),
            "energy_eV": energy,
            "rel_energy_meV": float((energy - e0) * 1000.0),
            "mobile_force_axis_eVA": float(np.dot(mobile_force, unit_c)),
            "mobile_force_norm_eVA": float(np.linalg.norm(mobile_force)),
            "max_force_eVA": float(np.linalg.norm(forces, axis=1).max()),
        }
        rows.append(row)
        print(
            f"{row['surface']} delta={delta_a:.4f} A "
            f"({row['delta_over_layer_spacing']:.4f} layer) "
            f"Erel={row['rel_energy_meV']:.6f} meV "
            f"Faxis={row['mobile_force_axis_eVA']:.6f} eV/A"
        )

        if args.write_structures:
            write(structures_dir / f"{args.name}_delta_{delta_a:.4f}A.POSCAR", atoms, format="vasp", direct=True, vasp5=True)

    csv_path = out_dir / f"{args.name}_scan.csv"
    md_path = out_dir / f"{args.name}_summary.md"
    png_abs_path = out_dir / f"{args.name}_energy_vs_delta_A.png"
    png_rel_path = out_dir / f"{args.name}_energy_vs_layer_spacing.png"
    png_force_path = out_dir / f"{args.name}_mobile_atom_force.png"
    html_path = out_dir / f"{args.name}_report.html"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plotted_abs = maybe_plot_energy(
        png_abs_path,
        rows,
        "delta_A",
        "absolute displacement along cell c (A)",
        f"{args.name}: energy vs absolute displacement",
    ) if args.plot else False
    plotted_rel = maybe_plot_energy(
        png_rel_path,
        rows,
        "delta_over_layer_spacing",
        "displacement / nearest lower-layer spacing",
        f"{args.name}: energy vs normalized displacement",
    ) if args.plot else False
    plotted_force = maybe_plot_force(
        png_force_path,
        rows,
        f"{args.name}: force on moved atom",
    ) if args.plot else False

    energy_values = [float(row["rel_energy_meV"]) for row in rows]
    force_axis_values = [float(row["mobile_force_axis_eVA"]) for row in rows]
    final_row = rows[-1]
    min_row = min(rows, key=lambda row: float(row["rel_energy_meV"]))
    max_row = max(rows, key=lambda row: float(row["rel_energy_meV"]))
    monotonic_decreases = sum(
        1
        for prev, curr in zip(energy_values, energy_values[1:])
        if curr < prev - 1e-6
    )

    summary_items = [
        f"Surface: `{args.surface or args.poscar.parent.name}`",
        f"Model: `{args.model.resolve()}`",
        f"POSCAR: `{args.poscar.resolve()}`",
        f"Device: `{args.device}`",
        f"dtype: `{args.default_dtype}`",
        f"Mobile atom: {mobile_element}{mobile_index + 1} (0-based index {mobile_index})",
        f"Extra vacuum along cell c: {args.vacuum:.6f} A",
        f"Cell c length after vacuum: {c_len_after_vacuum:.6f} A",
        f"Nearest lower-layer spacing: {layer_spacing:.6f} A",
        f"Scan points: {len(rows)}",
        f"Scan range: {rows[0]['delta_A']:.6f} to {rows[-1]['delta_A']:.6f} A",
        f"Final relative energy: {final_row['rel_energy_meV']:.6f} meV",
        f"Energy range: {min(energy_values):.6f} to {max(energy_values):.6f} meV",
        f"Minimum at delta={min_row['delta_A']:.6f} A, Erel={min_row['rel_energy_meV']:.6f} meV",
        f"Maximum at delta={max_row['delta_A']:.6f} A, Erel={max_row['rel_energy_meV']:.6f} meV",
        f"Monotonic decrease count after stepping outward: {monotonic_decreases}",
        f"Final axis force: {force_axis_values[-1]:.6f} eV/A",
    ]

    lines = [
        "# PtNi Slab Mobile-Atom Distance Scan",
        "",
        *[f"- {item}" for item in summary_items],
        "",
        "## Outputs",
        "",
        f"- CSV: `{csv_path}`",
        f"- Absolute displacement plot: `{png_abs_path if plotted_abs else 'not generated'}`",
        f"- Layer-normalized displacement plot: `{png_rel_path if plotted_rel else 'not generated'}`",
        f"- Force plot: `{png_force_path if plotted_force else 'not generated'}`",
        f"- HTML report: `{html_path if args.html else 'not generated'}`",
        "",
        "## Interpretation Notes",
        "",
        "- The moved atom is displaced along the third POSCAR lattice vector, matching an increase in the Direct-coordinate z value.",
        "- The slab is not stretched when vacuum is added; only the third lattice vector length is increased.",
        "- Energies are single-point MACE energies relative to the first scan point after vacuum extension.",
        "- `delta_over_layer_spacing` uses the nearest lower atomic layer below the moved atom along the third lattice-vector direction.",
        "- A stable model should avoid unphysical oscillations, sudden drops, or force spikes as the atom is pulled away from the surface.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.html:
        write_html_report(
            html_path,
            f"{args.name}: mobile atom distance scan",
            summary_items,
            rows,
            [png_abs_path, png_rel_path, png_force_path],
        )

    print(f"CSV: {csv_path}")
    print(f"Summary: {md_path}")
    if plotted_abs:
        print(f"Plot: {png_abs_path}")
    if plotted_rel:
        print(f"Plot: {png_rel_path}")
    if plotted_force:
        print(f"Plot: {png_force_path}")
    if args.html:
        print(f"HTML: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
