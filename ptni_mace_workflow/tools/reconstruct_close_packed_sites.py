#!/usr/bin/env python3
"""Pure geometric close-packed site reconstruction for off-lattice structures.

This is not a reference-lattice binding method.  Candidate sites are generated
only from the current atomic point cloud using local close-packed geometry:

1. nearest-neighbor line continuation: q = 2*r_i - r_j
2. local parallelogram closure: q = r_j + r_k - r_i
3. tetrahedral completion from near-equilateral NN triangles

The resulting point cloud is clustered, occupied sites are removed, and the
remaining unoccupied sites are filtered by local coordination and NP/slab
boundary checks.  He atoms in the output are pseudo-sites for visualization.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np


@dataclass
class StructureData:
    comment: str
    lattice: np.ndarray
    species: list[str]
    counts: list[int]
    atom_species: list[str]
    cart: np.ndarray


def parse_poscar(path: Path) -> StructureData:
    lines = [line.rstrip() for line in path.read_text().splitlines() if line.strip()]
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]) * scale
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    mode_index = 7
    if lines[mode_index].lower().startswith("s"):
        mode_index += 1
    direct = lines[mode_index].lower().startswith("d")
    natoms = sum(counts)
    raw = np.array([[float(x) for x in lines[mode_index + 1 + i].split()[:3]] for i in range(natoms)])
    cart = raw @ lattice if direct else raw * scale
    atom_species = []
    for sp, count in zip(species, counts):
        atom_species.extend([sp] * count)
    return StructureData(lines[0], lattice, species, counts, atom_species, cart)


def min_image(delta: np.ndarray, lattice: np.ndarray, inv_lattice: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    if not np.any(pbc):
        return delta
    frac = delta @ inv_lattice
    frac[pbc] -= np.round(frac[pbc])
    return frac @ lattice


def wrap_pbc(q: np.ndarray, lattice: np.ndarray, inv_lattice: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    if not np.any(pbc):
        return q
    frac = q @ inv_lattice
    frac[pbc] -= np.floor(frac[pbc])
    return frac @ lattice


def detect_mode(cart: np.ndarray, lattice: np.ndarray, requested: str) -> tuple[str, np.ndarray, dict]:
    lengths = np.linalg.norm(lattice, axis=1)
    span = cart.max(axis=0) - cart.min(axis=0)
    vacuum = lengths - span
    ratio = span / lengths
    if requested != "auto":
        mode = requested
    elif np.count_nonzero(vacuum > 5.0) >= 2 and np.all(ratio < 0.75):
        mode = "np"
    elif ratio[0] > 0.65 and ratio[1] > 0.65 and vacuum[2] > 5.0:
        mode = "slab"
    else:
        mode = "bulk"
    if mode == "slab":
        pbc = np.array([True, True, False])
    elif mode == "bulk":
        pbc = np.array([True, True, True])
    else:
        pbc = np.array([False, False, False])
    return mode, pbc, {
        "cell_lengths_A": lengths.tolist(),
        "cartesian_min_A": cart.min(axis=0).tolist(),
        "cartesian_max_A": cart.max(axis=0).tolist(),
        "cartesian_span_A": span.tolist(),
        "vacuum_estimate_A": vacuum.tolist(),
        "span_ratio": ratio.tolist(),
    }


def distance_matrix(cart: np.ndarray, lattice: np.ndarray, pbc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = np.linalg.inv(lattice)
    n = len(cart)
    vectors = np.zeros((n, n, 3), dtype=float)
    distances = np.zeros((n, n), dtype=float)
    for i in range(n):
        delta = cart - cart[i]
        if np.any(pbc):
            frac = delta @ inv
            frac[:, pbc] -= np.round(frac[:, pbc])
            delta = frac @ lattice
        vectors[i] = delta
        distances[i] = np.linalg.norm(delta, axis=1)
    np.fill_diagonal(distances, np.inf)
    return distances, vectors


def estimate_nn(distances: np.ndarray, cutoff: float) -> tuple[float, dict]:
    vals = distances[np.isfinite(distances)]
    shell = vals[(vals > 1.8) & (vals < cutoff)]
    if len(shell) < 20:
        shell = np.sort(vals[vals > 0.1])[: max(20, len(vals) // 20)]
    hist, edges = np.histogram(shell, bins=80)
    peak = int(np.argmax(hist))
    center = 0.5 * (edges[peak] + edges[peak + 1])
    narrow = shell[np.abs(shell - center) < 0.18]
    if len(narrow) < 10:
        narrow = shell
    d_nn = float(np.median(narrow))
    return d_nn, {
        "d_nn_estimate_A": d_nn,
        "a_fcc_estimate_A": d_nn * math.sqrt(2.0),
        "first_shell_pairs_count_directed": int(len(shell)),
        "first_shell_histogram_peak_A": float(center),
    }


def directional_hull_directions(points: np.ndarray, n_sphere: int = 768) -> np.ndarray:
    directions = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_sphere):
        z = 1.0 - 2.0 * (i + 0.5) / n_sphere
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * i
        directions.append([math.cos(theta) * r, math.sin(theta) * r, z])
    center = points.mean(axis=0)
    radial = points - center
    for vec in radial:
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            directions.append((vec / norm).tolist())
    arr = np.array(directions, dtype=float)
    arr /= np.linalg.norm(arr, axis=1)[:, None]
    _, idx = np.unique(np.round(arr, 8), axis=0, return_index=True)
    return arr[np.sort(idx)]


def make_boundary_filter(cart: np.ndarray, mode: str, boundary_tol: float) -> tuple[callable, dict]:
    if mode == "np":
        dirs = directional_hull_directions(cart)
        proj = cart @ dirs.T
        lo = proj.min(axis=0)
        hi = proj.max(axis=0)

        def inside(q: np.ndarray) -> bool:
            qp = q @ dirs.T
            return bool(np.all(qp >= lo - boundary_tol) and np.all(qp <= hi + boundary_tol))

        return inside, {"boundary_filter": "directional projection hull", "boundary_directions": int(len(dirs))}
    if mode == "slab":
        zmin = float(cart[:, 2].min())
        zmax = float(cart[:, 2].max())

        def inside(q: np.ndarray) -> bool:
            return bool(zmin - boundary_tol <= q[2] <= zmax + boundary_tol)

        return inside, {"boundary_filter": "slab z interval", "z_min_A": zmin, "z_max_A": zmax}

    def inside(_: np.ndarray) -> bool:
        return True

    return inside, {"boundary_filter": "periodic/no finite boundary"}


def add_candidate(raw: list[dict], q: np.ndarray, source: str, support: tuple[int, ...], quality: float) -> None:
    if np.all(np.isfinite(q)):
        raw.append({"position": q, "sources": {source}, "support": set(support), "quality_values": [float(quality)]})


def generate_candidates(
    data: StructureData,
    distances: np.ndarray,
    vectors: np.ndarray,
    d_nn: float,
    pbc: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    inv = np.linalg.inv(data.lattice)
    n = len(data.cart)
    lo = args.nn_low * d_nn
    hi = args.nn_high * d_nn
    nn = (distances > lo) & (distances < hi)
    raw: list[dict] = []

    # 1. Close-packed row continuation.
    for i in range(n):
        for j in np.where(nn[i])[0]:
            if i >= j:
                continue
            vij = vectors[i, j]
            add_candidate(raw, wrap_pbc(data.cart[i] - vij, data.lattice, inv, pbc), "line_extension", (i, j), abs(np.linalg.norm(vij) - d_nn))
            add_candidate(raw, wrap_pbc(data.cart[j] + vij, data.lattice, inv, pbc), "line_extension", (i, j), abs(np.linalg.norm(vij) - d_nn))

    # 2. Local fcc translation closure.  Fcc NN vector pairs have cosines near
    # -1/2, 0, or +1/2, excluding the opposite-vector case.
    allowed_cos = np.array([-0.5, 0.0, 0.5])
    for i in range(n):
        neigh = np.where(nn[i])[0]
        for a in range(len(neigh)):
            j = int(neigh[a])
            u = vectors[i, j]
            nu = np.linalg.norm(u)
            for b in range(a + 1, len(neigh)):
                k = int(neigh[b])
                v = vectors[i, k]
                nv = np.linalg.norm(v)
                cosang = float(np.dot(u, v) / (nu * nv))
                if np.min(np.abs(allowed_cos - cosang)) > args.angle_cos_tol:
                    continue
                q = data.cart[i] + u + v
                q = wrap_pbc(q, data.lattice, inv, pbc)
                add_candidate(raw, q, "parallelogram_closure", (i, j, k), np.min(np.abs(allowed_cos - cosang)))

    # 3. Tetrahedral completion from near-equilateral NN triangles.
    for i in range(n - 2):
        ni = set(np.where(nn[i])[0].tolist())
        for j in sorted(x for x in ni if x > i):
            common = sorted(k for k in ni.intersection(np.where(nn[j])[0].tolist()) if k > j)
            for k in common:
                dij = distances[i, j]
                dik = distances[i, k]
                djk = distances[j, k]
                sides = np.array([dij, dik, djk])
                rel_rms = float(np.sqrt(np.mean((sides - d_nn) ** 2)) / d_nn)
                if rel_rms > args.triangle_rel_tol:
                    continue
                a = data.cart[i]
                b = a + vectors[i, j]
                c = a + vectors[i, k]
                centroid = (a + b + c) / 3.0
                normal = np.cross(b - a, c - a)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal /= norm
                dbar = float(np.mean(sides))
                height_sq = dbar * dbar - (dbar / math.sqrt(3.0)) ** 2
                if height_sq <= 0:
                    continue
                height = math.sqrt(height_sq)
                for sign in (-1.0, 1.0):
                    q = wrap_pbc(centroid + sign * height * normal, data.lattice, inv, pbc)
                    add_candidate(raw, q, "tetrahedral_completion", (i, j, k), rel_rms)

    return raw


def cluster_candidates(raw: list[dict], tol: float) -> list[dict]:
    clusters: list[dict] = []
    buckets: dict[tuple[int, int, int], list[int]] = {}
    shifts = list(product((-1, 0, 1), repeat=3))
    for item in raw:
        q = item["position"]
        cell = tuple(np.floor(q / tol).astype(int).tolist())
        match = None
        for shift in shifts:
            key = (cell[0] + shift[0], cell[1] + shift[1], cell[2] + shift[2])
            for ci in buckets.get(key, []):
                if np.linalg.norm(q - clusters[ci]["position"]) < tol:
                    match = ci
                    break
            if match is not None:
                break
        if match is None:
            clusters.append(
                {
                    "position": q.copy(),
                    "positions": [q.copy()],
                    "sources": set(item["sources"]),
                    "support": set(item["support"]),
                    "quality_values": list(item["quality_values"]),
                    "votes": 1,
                }
            )
            buckets.setdefault(cell, []).append(len(clusters) - 1)
        else:
            c = clusters[match]
            c["positions"].append(q.copy())
            c["position"] = np.mean(c["positions"], axis=0)
            c["sources"].update(item["sources"])
            c["support"].update(item["support"])
            c["quality_values"].extend(item["quality_values"])
            c["votes"] += 1
    return clusters


def merge_close_clusters(clusters: list[dict], tol: float) -> list[dict]:
    """Iteratively merge cluster centers closer than tol.

    The first pass clusters streaming raw candidates, but a centroid can drift
    after later votes are added.  This pass rebuilds the neighborhood relation
    from final centers until no too-close centers remain.
    """
    current = clusters
    while True:
        merged: list[dict] = []
        used = [False] * len(current)
        changed = False
        order = sorted(range(len(current)), key=lambda i: current[i]["votes"], reverse=True)
        for idx in order:
            if used[idx]:
                continue
            base = _copy_cluster(current[idx])
            used[idx] = True
            absorbed = True
            while absorbed:
                absorbed = False
                for j in order:
                    if used[j]:
                        continue
                    if np.linalg.norm(current[j]["position"] - base["position"]) < tol:
                        _absorb_cluster(base, current[j])
                        used[j] = True
                        absorbed = True
                        changed = True
            merged.append(base)
        current = merged
        if not changed:
            return current


def _copy_cluster(cluster: dict) -> dict:
    return {
        "position": cluster["position"].copy(),
        "positions": [p.copy() for p in cluster["positions"]],
        "sources": set(cluster["sources"]),
        "support": set(cluster["support"]),
        "quality_values": list(cluster["quality_values"]),
        "votes": int(cluster["votes"]),
    }


def _absorb_cluster(target: dict, other: dict) -> None:
    target["positions"].extend(p.copy() for p in other["positions"])
    target["position"] = np.mean(np.array(target["positions"]), axis=0)
    target["sources"].update(other["sources"])
    target["support"].update(other["support"])
    target["quality_values"].extend(other["quality_values"])
    target["votes"] += int(other["votes"])


def classify_sites(
    clusters: list[dict],
    data: StructureData,
    d_nn: float,
    pbc: np.ndarray,
    inside_boundary: callable,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    inv = np.linalg.inv(data.lattice)
    kept: list[dict] = []
    discarded = {
        "occupied_by_real_atom": 0,
        "not_nn_shell_site": 0,
        "low_close_packed_coordination": 0,
        "outside_boundary": 0,
    }
    shell_lo = args.site_shell_low * d_nn
    shell_hi = args.site_shell_high * d_nn
    for c in clusters:
        q = c["position"]
        deltas = np.array([min_image(r - q, data.lattice, inv, pbc) for r in data.cart])
        d = np.linalg.norm(deltas, axis=1)
        min_d = float(d.min())
        if min_d < args.occupied_radius:
            discarded["occupied_by_real_atom"] += 1
            continue
        shell = (d > shell_lo) & (d < shell_hi)
        z = int(np.count_nonzero(shell))
        if z < args.z_min:
            discarded["low_close_packed_coordination"] += 1
            continue
        if min_d < shell_lo or min_d > shell_hi:
            discarded["not_nn_shell_site"] += 1
            continue
        if args.np_boundary == "none" and args.detected_mode != "slab":
            boundary_ok = True
        elif args.np_boundary == "one-shell" and args.detected_mode == "np":
            boundary_ok = inside_boundary(q) or c["votes"] >= args.min_votes
        else:
            boundary_ok = inside_boundary(q)
        if not boundary_ok:
            discarded["outside_boundary"] += 1
            continue
        shell_dist = d[shell]
        rms = float(np.sqrt(np.mean((shell_dist - d_nn) ** 2)))
        score = rms - 0.02 * min(c["votes"], 20) - 0.03 * min(z, 12)
        kept.append(
            {
                "index": len(kept) + 1,
                "cartesian_A": q.tolist(),
                "fractional": (q @ inv).tolist(),
                "coordination_in_nn_shell": z,
                "min_distance_to_real_atom_A": min_d,
                "nn_shell_rms_error_A": rms,
                "votes": int(c["votes"]),
                "support_atom_count": int(len(c["support"])),
                "sources": sorted(c["sources"]),
                "score": float(score),
            }
        )
    kept.sort(key=lambda s: (s["score"], -s["votes"], -s["coordination_in_nn_shell"]))
    for i, s in enumerate(kept, 1):
        s["index"] = i
    return kept, discarded


def write_poscar(data: StructureData, sites: list[dict], path: Path) -> None:
    lattice = data.lattice
    inv = np.linalg.inv(lattice)
    species = data.species + (["He"] if sites else [])
    counts = data.counts + ([len(sites)] if sites else [])
    coords = [*data.cart, *[np.array(s["cartesian_A"]) for s in sites]]
    frac = np.array(coords) @ inv
    lines = [f"{data.comment} | local close-packed feasible He sites", "1.0"]
    for row in lattice:
        lines.append("  " + "  ".join(f"{x:18.12f}" for x in row))
    lines.append("  " + "  ".join(species))
    lines.append("  " + "  ".join(str(x) for x in counts))
    lines.append("Direct")
    for row in frac:
        lines.append("  " + "  ".join(f"{x:18.12f}" for x in row))
    path.write_text("\n".join(lines) + "\n")


def write_xyz(data: StructureData, sites: list[dict], path: Path, local_only: bool, display_radius: float) -> None:
    real_indices = range(len(data.cart))
    if local_only and sites:
        keep = set()
        for s in sites:
            q = np.array(s["cartesian_A"])
            d = np.linalg.norm(data.cart - q, axis=1)
            keep.update(np.where(d < display_radius)[0].tolist())
        real_indices = sorted(keep)
    rows = [(data.atom_species[i], *data.cart[i]) for i in real_indices]
    rows.extend(("He", *s["cartesian_A"]) for s in sites)
    lines = [str(len(rows)), "close-packed feasible He sites"]
    for sp, x, y, z in rows:
        lines.append(f"{sp:2s} {x:16.8f} {y:16.8f} {z:16.8f}")
    path.write_text("\n".join(lines) + "\n")


def write_report(summary: dict, sites: list[dict], path: Path) -> None:
    lines = [
        "# Local Close-Packed Site Reconstruction",
        "",
        "This run uses local geometric closure only. It does not bind atoms to a global reference lattice.",
        "",
        "## Summary",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Real atoms: {summary['real_atom_count']}",
        f"- Estimated nearest-neighbor distance: {summary['d_nn_estimate_A']:.4f} A",
        f"- Estimated fcc lattice constant: {summary['a_fcc_estimate_A']:.4f} A",
        f"- Raw mathematical candidates: {summary['raw_candidate_count']}",
        f"- Clustered candidates: {summary['clustered_candidate_count']}",
        f"- Kept unoccupied close-packed sites: {summary['kept_site_count']}",
        f"- Boundary filter: {summary['boundary_filter']}",
        f"- NP boundary policy: {summary['np_boundary_policy']}",
        "",
        "## Kept Sites",
        "",
    ]
    if not sites:
        lines.append("No unoccupied close-packed sites passed the filters.")
    else:
        lines.append("| # | x | y | z | z_nn | min d(real) | votes | sources |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
        for s in sites[:100]:
            x, y, z = s["cartesian_A"]
            lines.append(
                f"| {s['index']} | {x:.4f} | {y:.4f} | {z:.4f} | "
                f"{s['coordination_in_nn_shell']} | {s['min_distance_to_real_atom_A']:.4f} | "
                f"{s['votes']} | {', '.join(s['sources'])} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--prefix", default="close_packed_sites")
    parser.add_argument("--mode", choices=["auto", "np", "slab", "bulk"], default="auto")
    parser.add_argument("--nn-cutoff", type=float, default=3.2)
    parser.add_argument("--nn-low", type=float, default=0.82)
    parser.add_argument("--nn-high", type=float, default=1.18)
    parser.add_argument("--site-shell-low", type=float, default=0.72)
    parser.add_argument("--site-shell-high", type=float, default=1.22)
    parser.add_argument("--occupied-radius", type=float, default=1.20)
    parser.add_argument("--z-min", type=int, default=3)
    parser.add_argument("--triangle-rel-tol", type=float, default=0.16)
    parser.add_argument("--angle-cos-tol", type=float, default=0.18)
    parser.add_argument("--cluster-tol", type=float, default=0.38)
    parser.add_argument("--merge-tol", type=float, default=0.80)
    parser.add_argument("--boundary-tol", type=float, default=0.35)
    parser.add_argument("--np-boundary", choices=["strict-hull", "one-shell", "none"], default="strict-hull")
    parser.add_argument("--min-votes", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = parse_poscar(args.input)
    mode, pbc, mode_info = detect_mode(data.cart, data.lattice, args.mode)
    args.detected_mode = mode
    distances, vectors = distance_matrix(data.cart, data.lattice, pbc)
    d_nn, nn_info = estimate_nn(distances, args.nn_cutoff)
    raw = generate_candidates(data, distances, vectors, d_nn, pbc, args)
    clusters = cluster_candidates(raw, args.cluster_tol)
    initial_cluster_count = len(clusters)
    clusters = merge_close_clusters(clusters, args.merge_tol)
    inside, boundary_info = make_boundary_filter(data.cart, mode, args.boundary_tol)
    sites, discarded = classify_sites(clusters, data, d_nn, pbc, inside, args)

    summary = {
        "input": str(args.input),
        "mode": mode,
        "pbc_axes_xyz": pbc.tolist(),
        "real_atom_count": len(data.cart),
        "raw_candidate_count": len(raw),
        "initial_clustered_candidate_count": initial_cluster_count,
        "clustered_candidate_count": len(clusters),
        "kept_site_count": len(sites),
        "discarded_counts": discarded,
        "occupied_radius_A": args.occupied_radius,
        "site_shell_low_A": args.site_shell_low * d_nn,
        "site_shell_high_A": args.site_shell_high * d_nn,
        "z_min": args.z_min,
        "cluster_tol_A": args.cluster_tol,
        "merge_tol_A": args.merge_tol,
        "np_boundary_policy": args.np_boundary,
        "min_votes": args.min_votes,
        **mode_info,
        **nn_info,
        **boundary_info,
    }

    base = args.output_dir / args.prefix
    write_poscar(data, sites, base.with_name(base.name + "_with_He.vasp"))
    write_xyz(data, sites, base.with_name(base.name + "_with_He.xyz"), local_only=False, display_radius=args.nn_cutoff)
    write_xyz(data, sites, base.with_name(base.name + "_local_display.xyz"), local_only=True, display_radius=args.nn_cutoff)
    (base.with_name(base.name + "_summary.json")).write_text(json.dumps({"summary": summary, "sites": sites}, indent=2))
    write_report(summary, sites, base.with_name(base.name + "_report.md"))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
