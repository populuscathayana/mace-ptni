#!/usr/bin/env python
"""Close-packed vacancy-site helpers for vacancy-mediated MCMD."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VacancySite:
    """One reconstructed or user-provided vacancy site."""

    site_index0: int
    source_index: int
    cartesian: np.ndarray
    fractional: np.ndarray | None
    coordination: int | None
    score: float | None
    raw: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        x, y, z = self.cartesian
        return {
            "site_index0": self.site_index0,
            "source_index": self.source_index,
            "x_A": float(x),
            "y_A": float(y),
            "z_A": float(z),
            "coordination": self.coordination,
            "score": self.score,
        }


@dataclass
class SiteReconstruction:
    summary: dict[str, Any]
    sites: list[VacancySite]
    structure_path: Path | None
    summary_path: Path | None
    he_poscar_path: Path | None


def _site_args(args: argparse.Namespace) -> argparse.Namespace:
    """Map MCMD CLI options to reconstruct_close_packed_sites arguments."""

    return argparse.Namespace(
        mode=args.site_mode,
        nn_cutoff=args.site_nn_cutoff,
        nn_low=args.site_nn_low,
        nn_high=args.site_nn_high,
        site_shell_low=args.site_shell_low,
        site_shell_high=args.site_shell_high,
        occupied_radius=args.site_occupied_radius,
        z_min=args.site_z_min,
        triangle_rel_tol=args.site_triangle_rel_tol,
        angle_cos_tol=args.site_angle_cos_tol,
        cluster_tol=args.site_cluster_tol,
        merge_tol=args.site_merge_tol,
        boundary_tol=args.site_boundary_tol,
        np_boundary=args.site_np_boundary,
        min_votes=args.site_min_votes,
    )


def _structure_data_from_atoms(atoms: Any, comment: str):
    """Create reconstruct_close_packed_sites.StructureData without a POSCAR round trip."""

    from collections import Counter

    from ptni_mace_workflow.tools import reconstruct_close_packed_sites as rcps

    symbols = atoms.get_chemical_symbols()
    counts_by_symbol = Counter(symbols)
    species: list[str] = []
    for symbol in symbols:
        if symbol not in species:
            species.append(symbol)
    counts = [counts_by_symbol[symbol] for symbol in species]
    return rcps.StructureData(
        comment=comment,
        lattice=np.asarray(atoms.cell.array, dtype=float),
        species=species,
        counts=counts,
        atom_species=symbols,
        cart=np.asarray(atoms.get_positions(), dtype=float),
    )


def reconstruct_sites_from_atoms(
    atoms: Any,
    output_dir: Path,
    prefix: str,
    args: argparse.Namespace,
) -> SiteReconstruction:
    """Run the close-packed site reconstruction on an ASE Atoms object."""

    from ase.io import write
    from ptni_mace_workflow.tools import reconstruct_close_packed_sites as rcps

    output_dir.mkdir(parents=True, exist_ok=True)
    structure_path: Path | None = None
    if args.site_output == "full":
        structure_path = output_dir / f"{prefix}_structure.vasp"
        write(structure_path.as_posix(), atoms, format="vasp", direct=True, vasp5=True)

    data = _structure_data_from_atoms(atoms, prefix)
    site_args = _site_args(args)
    mode, pbc, mode_info = rcps.detect_mode(data.cart, data.lattice, site_args.mode)
    site_args.detected_mode = mode
    distances, vectors = rcps.distance_matrix(data.cart, data.lattice, pbc)
    d_nn, nn_info = rcps.estimate_nn(distances, site_args.nn_cutoff)
    raw = rcps.generate_candidates(data, distances, vectors, d_nn, pbc, site_args)
    clusters = rcps.cluster_candidates(raw, site_args.cluster_tol)
    initial_cluster_count = len(clusters)
    clusters = rcps.merge_close_clusters(clusters, site_args.merge_tol)
    inside, boundary_info = rcps.make_boundary_filter(data.cart, mode, site_args.boundary_tol)
    raw_sites, discarded = rcps.classify_sites(clusters, data, d_nn, pbc, inside, site_args)

    summary = {
        "input": str(structure_path) if structure_path is not None else f"ase_atoms:{prefix}",
        "mode": mode,
        "pbc_axes_xyz": pbc.tolist(),
        "real_atom_count": len(data.cart),
        "raw_candidate_count": len(raw),
        "initial_clustered_candidate_count": initial_cluster_count,
        "clustered_candidate_count": len(clusters),
        "kept_site_count": len(raw_sites),
        "discarded_counts": discarded,
        "occupied_radius_A": site_args.occupied_radius,
        "site_shell_low_A": site_args.site_shell_low * d_nn,
        "site_shell_high_A": site_args.site_shell_high * d_nn,
        "z_min": site_args.z_min,
        "cluster_tol_A": site_args.cluster_tol,
        "merge_tol_A": site_args.merge_tol,
        "np_boundary_policy": site_args.np_boundary,
        "min_votes": site_args.min_votes,
        **mode_info,
        **nn_info,
        **boundary_info,
    }

    base = output_dir / prefix
    he_poscar_path: Path | None = None
    summary_path: Path | None = None

    if args.site_output in {"full", "vasp"}:
        he_poscar_path = base.with_name(base.name + "_with_He.vasp")
        rcps.write_poscar(data, raw_sites, he_poscar_path)

    if args.site_output == "full":
        he_xyz_path = base.with_name(base.name + "_with_He.xyz")
        local_xyz_path = base.with_name(base.name + "_local_display.xyz")
        summary_path = base.with_name(base.name + "_summary.json")
        report_path = base.with_name(base.name + "_report.md")
        rcps.write_xyz(data, raw_sites, he_xyz_path, local_only=False, display_radius=site_args.nn_cutoff)
        rcps.write_xyz(data, raw_sites, local_xyz_path, local_only=True, display_radius=site_args.nn_cutoff)
        import json

        summary_path.write_text(
            json.dumps({"summary": summary, "sites": raw_sites}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rcps.write_report(summary, raw_sites, report_path)

    inv = np.linalg.inv(np.asarray(atoms.cell.array, dtype=float))
    sites = [
        VacancySite(
            site_index0=i,
            source_index=int(site.get("index", i + 1)),
            cartesian=np.asarray(site["cartesian_A"], dtype=float),
            fractional=np.asarray(site.get("fractional", np.asarray(site["cartesian_A"]) @ inv), dtype=float),
            coordination=int(site["coordination_in_nn_shell"]) if site.get("coordination_in_nn_shell") is not None else None,
            score=float(site["score"]) if site.get("score") is not None else None,
            raw=dict(site),
        )
        for i, site in enumerate(raw_sites)
    ]

    return SiteReconstruction(
        summary=summary,
        sites=sites,
        structure_path=structure_path,
        summary_path=summary_path,
        he_poscar_path=he_poscar_path,
    )


def vacancy_from_cartesian(atoms: Any, cartesian: list[float] | tuple[float, float, float]) -> VacancySite:
    cart = np.asarray(cartesian, dtype=float)
    frac = cart @ np.linalg.inv(np.asarray(atoms.cell.array, dtype=float))
    return VacancySite(
        site_index0=-1,
        source_index=-1,
        cartesian=cart,
        fractional=frac,
        coordination=None,
        score=None,
        raw={"source": "user_cartesian"},
    )


def select_initial_vacancy(
    reconstruction: SiteReconstruction,
    atoms: Any,
    args: argparse.Namespace,
) -> tuple[VacancySite, str]:
    if args.vacancy_cartesian is not None:
        return vacancy_from_cartesian(atoms, args.vacancy_cartesian), "user_cartesian"

    if args.vacancy_site_index is not None:
        index = int(args.vacancy_site_index)
        if index < 0 or index >= len(reconstruction.sites):
            raise ValueError(
                f"--vacancy-site-index {index} is out of range for {len(reconstruction.sites)} reconstructed sites"
            )
        return reconstruction.sites[index], "user_site_index0"

    if args.auto_vacancy == "highest-score":
        if not reconstruction.sites:
            raise ValueError("auto vacancy selection requested, but no reconstructed sites were kept")
        return reconstruction.sites[0], "auto_highest_score"

    raise ValueError("Provide --vacancy-site-index, --vacancy-cartesian, or --auto-vacancy highest-score.")


def match_vacancy_to_reconstructed_site(
    previous_cartesian: np.ndarray,
    reconstruction: SiteReconstruction,
    max_distance: float,
) -> tuple[VacancySite, float, bool]:
    """Match the tracked vacancy to the nearest newly reconstructed site."""

    if not reconstruction.sites:
        fallback = VacancySite(
            site_index0=-1,
            source_index=-1,
            cartesian=np.asarray(previous_cartesian, dtype=float),
            fractional=None,
            coordination=None,
            score=None,
            raw={"source": "previous_cartesian_no_reconstructed_sites"},
        )
        return fallback, float("nan"), False

    distances = [float(np.linalg.norm(site.cartesian - previous_cartesian)) for site in reconstruction.sites]
    best_index = int(np.argmin(distances))
    best_distance = distances[best_index]
    if best_distance <= max_distance:
        return reconstruction.sites[best_index], best_distance, True

    fallback = VacancySite(
        site_index0=-1,
        source_index=-1,
        cartesian=np.asarray(previous_cartesian, dtype=float),
        fractional=None,
        coordination=None,
        score=None,
        raw={"source": "previous_cartesian_match_failed", "nearest_site_distance_A": best_distance},
    )
    return fallback, best_distance, False
