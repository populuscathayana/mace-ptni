#!/usr/bin/env python
"""Path helpers for the PtNi MACE workflow.

The workflow keeps source code and historical outputs separate from the new
isolated runtime workspace.  These helpers intentionally avoid importing MACE,
ASE, or torch so they can be used by lightweight migration and reporting tools.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


WORKSPACE_ENV = "MACE_WORKSPACE"


def repo_root(start: Path | None = None) -> Path:
    """Return the checkout root containing ``ptni_mace_workflow``."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "ptni_mace_workflow").is_dir():
            return candidate
    return Path.cwd().resolve()


def workspace_root(workspace: str | Path | None = None) -> Path:
    """Resolve the isolated runtime workspace.

    Priority: explicit argument, ``MACE_WORKSPACE`` environment variable, then
    ``<repo>/mace_workspace``.
    """
    if workspace is None:
        workspace = os.environ.get(WORKSPACE_ENV)
    path = Path(workspace) if workspace else repo_root() / "mace_workspace"
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def dataset_dir(dataset: str, workspace: str | Path | None = None) -> Path:
    return workspace_root(workspace) / "datasets" / dataset


def model_path(model_tag: str, workspace: str | Path | None = None) -> Path:
    return workspace_root(workspace) / "models" / model_tag / "model.model"


def training_run_dir(run_name: str, workspace: str | Path | None = None) -> Path:
    return workspace_root(workspace) / "runs" / "training" / run_name


def evaluation_run_dir(
    model_tag: str, dataset: str, workspace: str | Path | None = None
) -> Path:
    return workspace_root(workspace) / "runs" / "evaluation" / model_tag / dataset


def benchmark_run_dir(
    benchmark: str, model_tag: str, workspace: str | Path | None = None
) -> Path:
    return workspace_root(workspace) / "runs" / "benchmarks" / benchmark / model_tag


def ensure_workspace_layout(workspace: str | Path | None = None) -> Path:
    root = workspace_root(workspace)
    for rel in [
        "datasets/manifests",
        "inputs/pt111",
        "inputs/strained_neb",
        "inputs/np_structures",
        "models/ft_best_loss",
        "models/scratch_best_loss",
        "models/ft_np_baseline",
        "runs/training",
        "runs/evaluation",
        "runs/benchmarks",
        "reports/docs_site",
        "tmp",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_manifest_base(kind: str, name: str, workspace: str | Path | None = None) -> dict[str, Any]:
    root = workspace_root(workspace)
    return {
        "kind": kind,
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root()),
        "workspace": str(root),
    }
