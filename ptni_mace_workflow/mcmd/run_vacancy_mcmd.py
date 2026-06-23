#!/usr/bin/env python
"""Run a vacancy-mediated MCMD prototype with explicit CI-NEB barriers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from ptni_mace_workflow.common.mace_model import reject_checkpoint, resolve_model
from ptni_mace_workflow.common.paths import ensure_workspace_layout, mcmd_run_dir, run_manifest_base, write_json
from ptni_mace_workflow.mcmd.cache import event_dir, load_event_summary
from ptni_mace_workflow.mcmd.events import estimate_nn_from_atoms, generate_hop_events
from ptni_mace_workflow.mcmd.md import run_md_segment
from ptni_mace_workflow.mcmd.neb import NebResult, max_force, neb_result_from_summary, run_event_neb
from ptni_mace_workflow.mcmd.reports import append_csv_row, write_markdown_summary
from ptni_mace_workflow.mcmd.sites import (
    VacancySite,
    match_vacancy_to_reconstructed_site,
    reconstruct_sites_from_atoms,
    select_initial_vacancy,
)


EVENT_FIELDS = [
    "mcmd_step",
    "event_rank",
    "selected",
    "selection_probability",
    "rate_s^-1",
    "event_id",
    "atom_index0",
    "atom_index1",
    "atom_symbol",
    "vacancy_site_index0",
    "vacancy_source_index",
    "hop_distance_A",
    "direct_distance_A",
    "mic_distance_A",
    "crosses_pbc",
    "d_nn_A",
    "neb_status",
    "neb_converged",
    "cache_hit",
    "barrier_eV",
    "reverse_barrier_eV",
    "reaction_energy_eV",
    "max_force_eVA",
    "neb_dir",
    "summary_path",
    "message",
]

STEP_FIELDS = [
    "mcmd_step",
    "status",
    "selected_event_id",
    "atom_index0",
    "atom_index1",
    "atom_symbol",
    "barrier_eV",
    "reverse_barrier_eV",
    "reaction_energy_eV",
    "selected_rate_s^-1",
    "total_rate_s^-1",
    "selected_probability",
    "delta_time_s",
    "cumulative_time_s",
    "accepted_energy_eV",
    "accepted_fmax_eVA",
    "candidate_event_count",
    "total_candidate_event_count",
    "valid_rate_event_count",
    "vacancy_match_distance_A",
    "vacancy_matched_to_reconstructed_site",
    "site_count",
    "message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True, help="Initial POSCAR/extxyz structure.")
    parser.add_argument("--input-format", default=None, help="ASE input format. Default lets ASE infer from filename.")
    parser.add_argument("--model", type=Path, default=None, help="Explicit exported .model path.")
    parser.add_argument("--model-tag", default=None, help="Model tag under mace_workspace/models/<tag>/model.model.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite known CSV/manifest outputs in an existing run directory.")

    parser.add_argument("--vacancy-site-index", type=int, default=None, help="Zero-based index in the reconstructed site list.")
    parser.add_argument("--vacancy-cartesian", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--prepare-sites-only",
        action="store_true",
        help="Only write step_0000 close-packed site files, then stop before loading MACE or running MCMD.",
    )
    parser.add_argument("--vacancy-match-radius", type=float, default=1.0)

    parser.add_argument("--temperature", type=float, default=800.0)
    parser.add_argument("--attempt-frequency", type=float, default=1.0e13)
    parser.add_argument("--mc-steps", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--require-neb-converged", action="store_true")

    parser.add_argument("--md-steps", type=int, default=0)
    parser.add_argument("--md-timestep-fs", type=float, default=1.0)
    parser.add_argument("--md-ensemble", choices=["langevin", "nve"], default="langevin")
    parser.add_argument(
        "--md-position",
        choices=["before", "after", "both", "none"],
        default="after",
        help="Where to run the MD relaxation segment relative to the accepted MC-NEB hop. Default: after.",
    )
    parser.add_argument("--md-friction-per-fs", type=float, default=0.01)
    parser.add_argument("--md-write-interval", type=int, default=25)
    parser.add_argument("--write-md-frames", action="store_true", help="Write intermediate MD frames into trajectory.extxyz.")

    parser.add_argument("--neb-images", type=int, default=5)
    parser.add_argument("--neb-steps", type=int, default=100)
    parser.add_argument("--neb-fmax", type=float, default=0.05)
    parser.add_argument("--neb-maxstep", type=float, default=0.05)
    parser.add_argument("--endpoint-relax-mode", choices=["none", "full"], default="none")
    parser.add_argument("--endpoint-fmax", type=float, default=0.03)
    parser.add_argument("--endpoint-steps", type=int, default=100)
    parser.add_argument("--endpoint-maxstep", type=float, default=0.05)
    parser.add_argument("--no-climb", dest="climb", action="store_false")
    parser.set_defaults(climb=True)
    parser.add_argument("--fire-downhill-check", action="store_true")
    parser.add_argument("--write-ase-trajectory", action="store_true")
    parser.add_argument("--write-ase-log", action="store_true")
    parser.add_argument("--neb-output", choices=["full", "compact", "none"], default="full")

    parser.add_argument("--hop-shell-low", type=float, default=0.70)
    parser.add_argument("--hop-shell-high", type=float, default=1.30)
    parser.add_argument("--max-events-per-step", type=int, default=None)
    parser.add_argument("--event-order", choices=["nearest", "random"], default="nearest")
    parser.add_argument("--allow-pbc-hop", action="store_true")
    parser.add_argument("--pbc-cross-tol", type=float, default=0.25)
    parser.add_argument("--pbc", choices=["from-input", "true", "false"], default="from-input")
    parser.add_argument("--wrap-scaled", action="store_true")
    parser.add_argument("--site-output", choices=["full", "vasp", "none"], default="full")
    parser.add_argument("--site-mode", choices=["auto", "np", "slab", "bulk"], default="auto")
    parser.add_argument("--site-nn-cutoff", type=float, default=3.2)
    parser.add_argument("--site-nn-low", type=float, default=0.82)
    parser.add_argument("--site-nn-high", type=float, default=1.18)
    parser.add_argument("--site-shell-low", type=float, default=0.72)
    parser.add_argument("--site-shell-high", type=float, default=1.22)
    parser.add_argument("--site-occupied-radius", type=float, default=1.20)
    parser.add_argument("--site-z-min", type=int, default=3)
    parser.add_argument("--site-triangle-rel-tol", type=float, default=0.16)
    parser.add_argument("--site-angle-cos-tol", type=float, default=0.18)
    parser.add_argument("--site-cluster-tol", type=float, default=0.38)
    parser.add_argument("--site-merge-tol", type=float, default=0.80)
    parser.add_argument("--site-boundary-tol", type=float, default=0.35)
    parser.add_argument(
        "--site-np-boundary",
        choices=["strict-hull", "one-shell", "none"],
        default="one-shell",
        help="NP site boundary policy. Default one-shell keeps the close-packed shell as well as internal vacancies.",
    )
    parser.add_argument("--site-min-votes", type=int, default=2)
    return parser.parse_args()


def read_atoms(path: Path, input_format: str | None) -> Any:
    from ase.io import read

    if input_format:
        return read(path.as_posix(), format=input_format)
    return read(path.as_posix())


def apply_geometry_options(atoms: Any, args: argparse.Namespace) -> Any:
    prepared = atoms.copy()
    if args.pbc == "true":
        prepared.pbc = (True, True, True)
    elif args.pbc == "false":
        prepared.pbc = (False, False, False)
    if args.wrap_scaled:
        scaled = prepared.get_scaled_positions(wrap=False)
        prepared.set_scaled_positions(np.mod(scaled, 1.0))
    return prepared


def clean_known_outputs(run_dir: Path) -> None:
    for rel in [
        "run_manifest.json",
        "mcmd_steps.csv",
        "events.csv",
        "md_steps.csv",
        "trajectory.extxyz",
        "summary.md",
    ]:
        path = run_dir / rel
        if path.exists():
            path.unlink()


def evaluate_structure(atoms: Any, calc: Any) -> tuple[float, float]:
    atoms.calc = calc
    try:
        energy = float(atoms.get_potential_energy())
        force = max_force(atoms)
        return energy, force
    finally:
        atoms.calc = None


def write_state(trajectory: Path, atoms: Any, step: int, label: str, energy: float | None = None, fmax: float | None = None) -> None:
    from ase.io import write

    image = atoms.copy()
    image.info["mcmd_step"] = step
    image.info["state_label"] = label
    if energy is not None:
        image.info["MACE_energy"] = float(energy)
    if fmax is not None:
        image.info["MACE_fmax_eVA"] = float(fmax)
    write(trajectory.as_posix(), image, format="extxyz", append=trajectory.exists())


def result_is_eligible(result: NebResult, args: argparse.Namespace) -> bool:
    if result.status != "ok" or not math.isfinite(result.barrier_eV):
        return False
    if args.require_neb_converged and not result.converged:
        return False
    return True


def select_result(results: list[NebResult], rates: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray, float]:
    total = float(np.sum(rates))
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("all event rates are zero or non-finite")
    probabilities = rates / total
    selected = int(rng.choice(len(results), p=probabilities))
    return selected, probabilities, total


def main() -> int:
    args = parse_args()
    workspace = ensure_workspace_layout(args.workspace)
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"input structure not found: {input_path}")
    if args.mc_steps < 0:
        raise SystemExit("--mc-steps must be >= 0")
    if args.md_steps < 0:
        raise SystemExit("--md-steps must be >= 0")

    run_dir = mcmd_run_dir(args.run_name, workspace)
    if run_dir.exists() and (run_dir / "mcmd_steps.csv").exists() and not args.overwrite:
        raise SystemExit(f"{run_dir} already contains mcmd_steps.csv. Use --overwrite or a new --run-name.")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clean_known_outputs(run_dir)

    site_dir = run_dir / "site_reports"
    cache_root = run_dir / "neb_cache"
    trajectory = run_dir / "trajectory.extxyz"
    events_csv = run_dir / "events.csv"
    steps_csv = run_dir / "mcmd_steps.csv"
    md_csv = run_dir / "md_steps.csv"
    summary_md = run_dir / "summary.md"

    atoms = apply_geometry_options(read_atoms(input_path, args.input_format), args)

    needs_initial_vacancy = args.vacancy_site_index is None and args.vacancy_cartesian is None
    if args.prepare_sites_only or needs_initial_vacancy:
        reconstruction = reconstruct_sites_from_atoms(atoms, site_dir, "step_0000", args)
        manifest = run_manifest_base("mcmd_site_prepare", args.run_name, workspace)
        manifest.update(
            {
                "input": str(input_path),
                "status": "needs_vacancy_selection",
                "site_count": len(reconstruction.sites),
                "site_report_vasp": str(reconstruction.he_poscar_path or ""),
                "site_summary_json": str(reconstruction.summary_path or ""),
                "message": "Inspect step_0000_with_He.vasp and rerun with --vacancy-site-index or --vacancy-cartesian.",
            }
        )
        write_json(run_dir / "run_manifest.json", manifest)
        print(f"Prepared initial close-packed sites: {reconstruction.he_poscar_path}")
        print(f"Site count: {len(reconstruction.sites)}")
        print("Rerun with --vacancy-site-index INDEX or --vacancy-cartesian X Y Z to start MCMD.")
        if args.prepare_sites_only:
            return 0
        raise SystemExit("Initial vacancy is required; inspect step_0000_with_He.vasp and rerun with an explicit vacancy.")

    model = resolve_model(args.model, args.model_tag, workspace)
    reject_checkpoint(model)
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")

    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=str(model),
        device=args.device,
        default_dtype=args.default_dtype,
    )

    initial_energy, initial_fmax = evaluate_structure(atoms, calc)
    write_state(trajectory, atoms, 0, "initial", initial_energy, initial_fmax)

    manifest = run_manifest_base("mcmd", args.run_name, workspace)
    manifest.update(
        {
            "input": str(input_path),
            "model": str(model),
            "model_tag": args.model_tag or "",
            "device": args.device,
            "default_dtype": args.default_dtype,
            "temperature_K": args.temperature,
            "attempt_frequency_s^-1": args.attempt_frequency,
            "mc_steps": args.mc_steps,
            "md_steps": args.md_steps,
            "md_timestep_fs": args.md_timestep_fs,
            "md_position": args.md_position,
            "neb_images": args.neb_images,
            "neb_steps": args.neb_steps,
            "neb_fmax": args.neb_fmax,
            "neb_output": args.neb_output,
            "event_order": args.event_order,
            "max_events_per_step": args.max_events_per_step,
            "endpoint_relax_mode": args.endpoint_relax_mode,
            "initial_energy_eV": initial_energy,
            "initial_fmax_eVA": initial_fmax,
            "status": "running",
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)

    rng = np.random.default_rng(args.random_seed)
    current_atoms = atoms.copy()
    current_vacancy: VacancySite | None = None
    cumulative_time = 0.0
    step_rows: list[dict[str, Any]] = []
    event_count = 0

    for step in range(args.mc_steps):
        if args.md_position in {"before", "both"}:
            current_atoms = run_md_segment(
                current_atoms,
                calc,
                step,
                args.temperature,
                args.md_steps,
                args.md_timestep_fs,
                args.md_ensemble,
                args.md_friction_per_fs,
                md_csv,
                trajectory if args.write_md_frames and args.md_steps > 0 else None,
                args.md_write_interval,
            )
            current_atoms.calc = None

        reconstruction = reconstruct_sites_from_atoms(current_atoms, site_dir, f"step_{step:04d}", args)
        if step == 0 and current_vacancy is None:
            current_vacancy, vacancy_source = select_initial_vacancy(reconstruction, current_atoms, args)
            match_distance = 0.0
            matched = True
        else:
            assert current_vacancy is not None
            matched_vacancy, match_distance, matched = match_vacancy_to_reconstructed_site(
                current_vacancy.cartesian,
                reconstruction,
                args.vacancy_match_radius,
            )
            if not matched or matched_vacancy is None:
                row = {
                    "mcmd_step": step,
                    "status": "stopped_vacancy_not_matched",
                    "candidate_event_count": 0,
                    "total_candidate_event_count": 0,
                    "valid_rate_event_count": 0,
                    "vacancy_match_distance_A": match_distance,
                    "vacancy_matched_to_reconstructed_site": matched,
                    "site_count": len(reconstruction.sites),
                    "message": "current vacancy could not be matched to a reconstructed close-packed site",
                }
                append_csv_row(steps_csv, row, STEP_FIELDS)
                step_rows.append(row)
                break
            current_vacancy = matched_vacancy
            vacancy_source = "matched_reconstructed_site"

        d_nn = float(reconstruction.summary.get("d_nn_estimate_A") or estimate_nn_from_atoms(current_atoms))
        all_events = generate_hop_events(
            current_atoms,
            current_vacancy,
            step,
            d_nn,
            args.hop_shell_low,
            args.hop_shell_high,
            args.allow_pbc_hop,
            args.pbc_cross_tol,
            None,
        )
        if args.event_order == "random" and all_events:
            order = rng.permutation(len(all_events))
            events = [all_events[int(index)] for index in order]
        else:
            events = list(all_events)
        if args.max_events_per_step is not None:
            events = events[: args.max_events_per_step]

        print(
            f"MC step {step}: sites={len(reconstruction.sites)} "
            f"vacancy=({current_vacancy.cartesian[0]:.4f}, {current_vacancy.cartesian[1]:.4f}, {current_vacancy.cartesian[2]:.4f}) "
            f"source={vacancy_source} candidate_events={len(events)}/{len(all_events)}",
            flush=True,
        )

        if not events:
            row = {
                "mcmd_step": step,
                "status": "stopped_no_events",
                "candidate_event_count": 0,
                "total_candidate_event_count": len(all_events),
                "valid_rate_event_count": 0,
                "vacancy_match_distance_A": match_distance,
                "vacancy_matched_to_reconstructed_site": matched,
                "site_count": len(reconstruction.sites),
                "message": "no candidate vacancy-mediated nearest-neighbor hop events",
            }
            append_csv_row(steps_csv, row, STEP_FIELDS)
            step_rows.append(row)
            break

        results: list[NebResult] = []
        for rank, event in enumerate(events):
            cached = load_event_summary(cache_root, event.event_id)
            if cached is not None:
                result = neb_result_from_summary(event, cached, cache_root)
            else:
                result = run_event_neb(event, calc, args, event_dir(cache_root, event.event_id))
            results.append(result)
            event_count += 1

        rates = np.array(
            [
                result.rate(args.temperature, args.attempt_frequency) if result_is_eligible(result, args) else 0.0
                for result in results
            ],
            dtype=float,
        )

        try:
            selected_index, probabilities, total_rate = select_result(results, rates, rng)
        except ValueError as exc:
            for rank, result in enumerate(results):
                event_row = {
                    "mcmd_step": step,
                    "event_rank": rank,
                    "selected": False,
                    "selection_probability": 0.0,
                    "rate_s^-1": rates[rank] if rank < len(rates) else 0.0,
                    **result.event.to_row(),
                    **result.to_row(),
                }
                append_csv_row(events_csv, event_row, EVENT_FIELDS)
            row = {
                "mcmd_step": step,
                "status": "stopped_no_valid_rates",
                "candidate_event_count": len(events),
                "total_candidate_event_count": len(all_events),
                "valid_rate_event_count": int(np.count_nonzero(rates > 0.0)),
                "vacancy_match_distance_A": match_distance,
                "vacancy_matched_to_reconstructed_site": matched,
                "site_count": len(reconstruction.sites),
                "message": str(exc),
            }
            append_csv_row(steps_csv, row, STEP_FIELDS)
            step_rows.append(row)
            break

        selected = results[selected_index]
        for rank, result in enumerate(results):
            event_row = {
                "mcmd_step": step,
                "event_rank": rank,
                "selected": rank == selected_index,
                "selection_probability": f"{probabilities[rank]:.12e}",
                "rate_s^-1": f"{rates[rank]:.12e}",
                **result.event.to_row(),
                **result.to_row(),
            }
            append_csv_row(events_csv, event_row, EVENT_FIELDS)

        delta_time = 1.0 / total_rate if total_rate > 0 else float("inf")
        cumulative_time += delta_time
        current_atoms = selected.final_atoms.copy()
        current_vacancy = VacancySite(
            site_index0=-1,
            source_index=-1,
            cartesian=selected.event.new_vacancy_cartesian.copy(),
            fractional=None,
            coordination=None,
            score=None,
            raw={"source": "post_hop_atom_old_position", "selected_event_id": selected.event.event_id},
        )

        if args.md_position in {"after", "both"}:
            current_atoms = run_md_segment(
                current_atoms,
                calc,
                step,
                args.temperature,
                args.md_steps,
                args.md_timestep_fs,
                args.md_ensemble,
                args.md_friction_per_fs,
                md_csv,
                trajectory if args.write_md_frames and args.md_steps > 0 else None,
                args.md_write_interval,
            )
            current_atoms.calc = None

        accepted_energy, accepted_fmax = evaluate_structure(current_atoms, calc)
        write_state(trajectory, current_atoms, step + 1, "accepted_mc_post_md", accepted_energy, accepted_fmax)

        row = {
            "mcmd_step": step,
            "status": "accepted",
            "selected_event_id": selected.event.event_id,
            "atom_index0": selected.event.atom_index,
            "atom_index1": selected.event.atom_index + 1,
            "atom_symbol": selected.event.atom_symbol,
            "barrier_eV": f"{selected.barrier_eV:.12f}",
            "reverse_barrier_eV": f"{selected.reverse_barrier_eV:.12f}",
            "reaction_energy_eV": f"{selected.reaction_energy_eV:.12f}",
            "selected_rate_s^-1": f"{rates[selected_index]:.12e}",
            "total_rate_s^-1": f"{total_rate:.12e}",
            "selected_probability": f"{probabilities[selected_index]:.12e}",
            "delta_time_s": f"{delta_time:.12e}",
            "cumulative_time_s": f"{cumulative_time:.12e}",
            "accepted_energy_eV": f"{accepted_energy:.12f}",
            "accepted_fmax_eVA": f"{accepted_fmax:.12f}",
            "candidate_event_count": len(events),
            "total_candidate_event_count": len(all_events),
            "valid_rate_event_count": int(np.count_nonzero(rates > 0.0)),
            "vacancy_match_distance_A": match_distance,
            "vacancy_matched_to_reconstructed_site": matched,
            "site_count": len(reconstruction.sites),
            "message": selected.message,
        }
        append_csv_row(steps_csv, row, STEP_FIELDS)
        step_rows.append(row)
        print(
            f"  selected {selected.event.event_id}: "
            f"Ea={selected.barrier_eV:.6f} eV "
            f"rate={rates[selected_index]:.4e} s^-1 "
            f"p={probabilities[selected_index]:.4f}",
            flush=True,
        )

    manifest["status"] = "complete"
    manifest["accepted_steps"] = sum(1 for row in step_rows if row.get("status") == "accepted")
    manifest["event_count"] = event_count
    manifest["cumulative_time_s"] = cumulative_time
    manifest["run_dir"] = str(run_dir)
    write_json(run_dir / "run_manifest.json", manifest)
    write_markdown_summary(summary_md, manifest, step_rows, event_count)

    print(f"Run directory: {run_dir}")
    print(f"Manifest: {run_dir / 'run_manifest.json'}")
    print(f"Steps: {steps_csv}")
    print(f"Events: {events_csv}")
    print(f"Summary: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
