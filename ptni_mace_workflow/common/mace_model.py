#!/usr/bin/env python
"""Model path conventions for benchmark and evaluation entrypoints."""

from __future__ import annotations

from pathlib import Path

from .paths import model_path


def resolve_model(model: str | Path | None, model_tag: str | None, workspace: str | Path | None) -> Path:
    if model is not None:
        path = Path(model)
        return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    if not model_tag:
        raise ValueError("Either --model or --model-tag is required.")
    return model_path(model_tag, workspace)


def reject_checkpoint(path: Path) -> None:
    if path.suffix == ".pt":
        raise ValueError(
            f"{path} looks like a training checkpoint. Use an exported .model file for evaluation."
        )
