#!/usr/bin/env python
"""Predict the fcc Pt equilibrium lattice constant with a MACE model.

The script scans conventional fcc lattice constants, computes MACE single-point
energies, fits the minimum, and writes CSV/Markdown/PNG outputs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_scan(text: str) -> list[float]:
    """Parse start:stop:step or comma-separated lattice constants."""
    if ":" in text:
        parts = [float(part) for part in text.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("scan must be start:stop:step")
        start, stop, step = parts
        if step <= 0:
            raise argparse.ArgumentTypeError("scan step must be positive")
        values = []
        current = start
        # Include stop when it lands on the grid, with a small tolerance.
        while current <= stop + step * 1e-9:
            values.append(round(current, 10))
            current += step
        return values
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def quadratic_fit_minimum(a_values: list[float], e_values: list[float]) -> tuple[float, float, tuple[float, float, float]]:
    import numpy as np

    coeff = np.polyfit(np.asarray(a_values), np.asarray(e_values), deg=2)
    c2, c1, c0 = [float(x) for x in coeff]
    if c2 <= 0:
        raise ValueError(f"quadratic fit is not convex: c2={c2}")
    a0 = -c1 / (2.0 * c2)
    e0 = c2 * a0 * a0 + c1 * a0 + c0
    return a0, e0, (c2, c1, c0)


def ase_eos_fit(a_values: list[float], e_values: list[float], cubic: bool) -> tuple[float, float, float]:
    from ase.eos import EquationOfState

    volumes = [(a**3 if cubic else a) for a in a_values]
    eos = EquationOfState(volumes, e_values, eos="birchmurnaghan")
    v0, e0, bulk_modulus = eos.fit()
    a0 = v0 ** (1.0 / 3.0) if cubic else v0
    return float(a0), float(e0), float(bulk_modulus)


def maybe_plot(path: Path, rows: list[dict], a0: float, e0: float, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    a_values = [row["a_A"] for row in rows]
    e_values = [row["energy_per_atom_eV"] for row in rows]
    e_min = min(e_values)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(a_values, [(e - e_min) * 1000.0 for e in e_values], marker="o")
    ax.axvline(a0, color="tab:red", linestyle="--", label=f"a0={a0:.5f} A")
    ax.scatter([a0], [(e0 - e_min) * 1000.0], color="tab:red", zorder=3)
    ax.set_xlabel("fcc Pt lattice constant a (A)")
    ax.set_ylabel("relative energy (meV/atom)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan fcc Pt lattice constants using MACE.")
    parser.add_argument("--model", type=Path, required=True, help="MACE .model file, not compiled.model.")
    parser.add_argument(
        "--scan",
        type=parse_scan,
        default=parse_scan("3.75:4.15:0.01"),
        help="Lattice constants in A. Use start:stop:step or comma list. Default: 3.75:4.15:0.01.",
    )
    parser.add_argument(
        "--fit-window",
        type=float,
        default=0.08,
        help="Use points within +/- this many A of scanned minimum for quadratic fit. Default: 0.08.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", default="float64")
    parser.add_argument(
        "--supercell",
        default="1,1,1",
        help="Repeat conventional fcc cell, e.g. 2,2,2. Energies are reported per atom.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("mace_workspace/runs/benchmarks/lattice/manual/pt_fcc_lattice_test"))
    parser.add_argument("--name", default="pt_fcc_mace")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    from ase.build import bulk
    from mace.calculators import MACECalculator

    reps = tuple(int(part) for part in args.supercell.split(","))
    if len(reps) != 3 or any(value < 1 for value in reps):
        raise SystemExit("--supercell must look like 1,1,1 with positive integers")

    calc = MACECalculator(
        model_paths=str(args.model.resolve()),
        device=args.device,
        default_dtype=args.default_dtype,
    )

    rows = []
    for a in args.scan:
        atoms = bulk("Pt", "fcc", a=a, cubic=True)
        if reps != (1, 1, 1):
            atoms = atoms.repeat(reps)
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        rows.append(
            {
                "a_A": a,
                "natoms": len(atoms),
                "energy_eV": energy,
                "energy_per_atom_eV": energy / len(atoms),
            }
        )
        print(f"a={a:.6f} A  E/atom={energy / len(atoms):.10f} eV")

    scanned_min = min(rows, key=lambda row: row["energy_per_atom_eV"])
    fit_rows = [
        row
        for row in rows
        if abs(row["a_A"] - scanned_min["a_A"]) <= args.fit_window + 1e-12
    ]
    if len(fit_rows) < 3:
        raise SystemExit("Need at least 3 points in fit window; increase --fit-window or scan density.")

    fit_a = [row["a_A"] for row in fit_rows]
    fit_e = [row["energy_per_atom_eV"] for row in fit_rows]
    quad_a0, quad_e0, quad_coeff = quadratic_fit_minimum(fit_a, fit_e)

    eos_status = "ok"
    try:
        eos_a0, eos_e0, eos_bulk_modulus = ase_eos_fit(fit_a, fit_e, cubic=True)
    except Exception as exc:
        eos_status = repr(exc)
        eos_a0 = eos_e0 = eos_bulk_modulus = None

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.name}_scan.csv"
    md_path = out_dir / f"{args.name}_summary.md"
    png_path = out_dir / f"{args.name}_curve.png"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plotted = maybe_plot(png_path, rows, quad_a0, quad_e0, "MACE fcc Pt lattice scan") if args.plot else False

    lines = [
        "# fcc Pt Lattice Constant Test",
        "",
        f"- Model: `{args.model.resolve()}`",
        f"- Device: `{args.device}`",
        f"- dtype: `{args.default_dtype}`",
        f"- Supercell: `{args.supercell}` conventional fcc cell repeats",
        f"- Scan points: {len(rows)}",
        f"- Scanned minimum: a={scanned_min['a_A']:.6f} A, E/atom={scanned_min['energy_per_atom_eV']:.10f} eV",
        f"- Quadratic fit a0: {quad_a0:.8f} A",
        f"- Quadratic fit E0/atom: {quad_e0:.10f} eV",
        f"- Quadratic coefficients c2,c1,c0: {quad_coeff}",
    ]
    if eos_a0 is not None:
        lines.extend(
            [
                f"- Birch-Murnaghan EOS a0: {eos_a0:.8f} A",
                f"- Birch-Murnaghan EOS E0/atom: {eos_e0:.10f} eV",
                f"- Birch-Murnaghan bulk modulus raw ASE units: {eos_bulk_modulus:.10f}",
            ]
        )
    else:
        lines.append(f"- Birch-Murnaghan EOS fit failed: {eos_status}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- CSV: `{csv_path}`",
            f"- Plot: `{png_path if plotted else 'not generated'}`",
            "",
            "## Notes",
            "",
            "- The reported lattice constant is the conventional cubic fcc lattice constant.",
            "- Compare against the same DFT functional/settings used in the training data, not only experimental room-temperature Pt.",
            "- This is a single-point energy curve test; it does not by itself validate surface, defect, or transition-state behavior.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"CSV: {csv_path}")
    print(f"Summary: {md_path}")
    if plotted:
        print(f"Plot: {png_path}")
    print(f"Quadratic a0: {quad_a0:.8f} A")
    if eos_a0 is not None:
        print(f"EOS a0: {eos_a0:.8f} A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
