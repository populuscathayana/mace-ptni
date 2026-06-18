#!/usr/bin/env python
"""Copy canonical inputs into ``mace_workspace`` and write a migration manifest.

This tool is intentionally conservative: it copies only the data that the new
workflow should treat as canonical, and it never deletes or moves historical
``work/`` or ``outputs/`` content.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ptni_mace_workflow.common.paths import (
    ensure_workspace_layout,
    repo_root,
    sha256_file,
    workspace_root,
)
from ptni_mace_workflow.common.reports import write_csv_rows


@dataclass(frozen=True)
class CopySpec:
    source: Path
    destination: Path
    label: str
    kind: str


def default_specs(workspace: Path) -> list[CopySpec]:
    root = repo_root()
    return [
        CopySpec(root / "work/datasets/ptni_split", workspace / "datasets/ptni_split", "ptni_split", "dataset"),
        CopySpec(
            root / "work/datasets/NP_benchmark_package",
            workspace / "datasets/NP_benchmark_package",
            "NP_benchmark_package",
            "dataset",
        ),
        CopySpec(root / "work/energy_surface_inputs/Pt111", workspace / "inputs/pt111", "Pt111 PES", "input"),
        CopySpec(root / "work/neb_inputs", workspace / "inputs/strained_neb", "strained NEB POSCARs", "input"),
        CopySpec(root / "work/NP_input", workspace / "inputs/np_structures", "manual NP relax inputs", "input"),
        CopySpec(
            root / "checkpoints/ptni_binary_mace_ft_run-123_best_loss.model",
            workspace / "models/ft_best_loss/model.model",
            "ft_best_loss",
            "model",
        ),
        CopySpec(
            root / "checkpoints/ptni_binary_mace_scratch_run-123_best_loss.model",
            workspace / "models/scratch_best_loss/model.model",
            "scratch_best_loss",
            "model",
        ),
        CopySpec(
            root / "checkpoints/ptni_binary_mace_ft_best_loss.model",
            workspace / "models/ft_np_baseline/model.model",
            "ft_np_baseline",
            "model",
        ),
    ]


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return []


def copy_spec(spec: CopySpec, force: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not spec.source.exists():
        rows.append(
            {
                "status": "missing_source",
                "kind": spec.kind,
                "label": spec.label,
                "source": str(spec.source),
                "destination": str(spec.destination),
                "size_bytes": "",
                "sha256": "",
                "copied_at": "",
            }
        )
        return rows

    if spec.source.is_file():
        pairs = [(spec.source, spec.destination)]
    else:
        files = iter_files(spec.source)
        pairs = [(src, spec.destination / src.relative_to(spec.source)) for src in files]

    copied_at = datetime.now().isoformat(timespec="seconds")
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if force or not dst.exists() or src.stat().st_size != dst.stat().st_size:
            shutil.copy2(src, dst)
            status = "copied"
        else:
            status = "kept_existing"
        rows.append(
            {
                "status": status,
                "kind": spec.kind,
                "label": spec.label,
                "source": str(src),
                "destination": str(dst),
                "size_bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
                "copied_at": copied_at,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=None, help="Workspace root. Default: MACE_WORKSPACE or mace_workspace.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing destination files.")
    args = parser.parse_args()

    workspace = ensure_workspace_layout(workspace_root(args.workspace))
    manifest = workspace / "datasets/manifests/migration_manifest.csv"

    rows: list[dict[str, object]] = []
    for spec in default_specs(workspace):
        rows.extend(copy_spec(spec, force=args.force))

    fieldnames = ["status", "kind", "label", "source", "destination", "size_bytes", "sha256", "copied_at"]
    write_csv_rows(manifest, rows, fieldnames)
    copied = sum(1 for row in rows if row["status"] == "copied")
    missing = sum(1 for row in rows if row["status"] == "missing_source")
    print(f"Workspace: {workspace}")
    print(f"Manifest: {manifest}")
    print(f"Rows: {len(rows)} copied={copied} missing={missing}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
