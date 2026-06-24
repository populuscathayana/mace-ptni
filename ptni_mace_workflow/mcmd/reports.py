#!/usr/bin/env python
"""CSV and Markdown reporting helpers for MCMD runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_markdown_summary(path: Path, manifest: dict[str, Any], step_rows: list[dict[str, Any]], event_count: int) -> None:
    lines = [
        "# Vacancy-Mediated MCMD Run",
        "",
        "This is an early MCMD prototype. MD is handled by ASE+MACE, MC hop events are vacancy-mediated nearest-neighbor moves, and event barriers are computed with explicit ASE CI-NEB.",
        "",
        "## Run",
        "",
        f"- Run name: `{manifest.get('name', '')}`",
        f"- Workspace: `{manifest.get('workspace', '')}`",
        f"- Input: `{manifest.get('input', '')}`",
        f"- Model: `{manifest.get('model', '')}`",
        f"- Device: `{manifest.get('device', '')}`",
        f"- Temperature: {manifest.get('temperature_K', '')} K",
        f"- MC steps requested: {manifest.get('mc_steps', '')}",
        f"- MD steps per MC step: {manifest.get('md_steps', '')}",
        f"- NEB images: {manifest.get('neb_images', '')}",
        f"- NEB fmax: {manifest.get('neb_fmax', '')} eV/A",
        "",
        "## Output Files",
        "",
        "- `run_manifest.json`: full settings and completion state.",
        "- `mcmd_steps.csv`: selected MC event for each accepted step.",
        "- `events.csv`: all candidate events and their NEB barriers.",
        "- `trajectory.extxyz`: accepted MC states and optional MD frames.",
        "- `site_reports/`: reconstructed vacancy-site reports.",
        "- `neb_cache/`: per-event NEB paths and energy profiles.",
        "",
        "## Selected Steps",
        "",
    ]

    if not step_rows:
        lines.append("No MC hop has been selected yet.")
    else:
        lines.append("| step | type | selected event | atom | coord | barrier eV | rate s^-1 | probability | status |")
        lines.append("|---:|---|---|---|---:|---:|---:|---:|---|")
        for row in step_rows:
            lines.append(
                f"| {row.get('mcmd_step', '')} | {row.get('event_type', '')} | `{row.get('selected_event_id', '')}` | "
                f"{row.get('atom_symbol', '')}{row.get('atom_index1', '')} | "
                f"{row.get('initial_coordination', '')}->{row.get('final_coordination_at_target', '')} | "
                f"{row.get('barrier_eV', '')} | {row.get('selected_rate_s^-1', '')} | "
                f"{row.get('selected_probability', '')} | {row.get('neb_status', '')} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Candidate events evaluated: {event_count}",
            "- `vacancy-site-index` is zero-based in the MCMD CLI, while the close-packed site report keeps its original one-based `source_index`.",
            "- By default, hop events that appear to cross a periodic boundary are skipped; use `--allow-pbc-hop` only after inspecting interpolation behavior.",
            "- CI-NEB climbing images use projected NEB forces, not zero force on the highest-energy image.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
