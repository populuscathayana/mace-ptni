#!/usr/bin/env python
"""Small JSON cache helpers for MCMD NEB events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def event_dir(cache_root: Path, event_id: str) -> Path:
    return cache_root / event_id


def summary_path(cache_root: Path, event_id: str) -> Path:
    return event_dir(cache_root, event_id) / "summary.json"


def load_event_summary(cache_root: Path, event_id: str) -> dict[str, Any] | None:
    path = summary_path(cache_root, event_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_event_summary(cache_root: Path, event_id: str, summary: dict[str, Any]) -> Path:
    path = summary_path(cache_root, event_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def cache_hit_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_hit": True,
        "neb_status": summary.get("status", ""),
        "neb_converged": summary.get("converged", ""),
        "barrier_eV": summary.get("barrier_eV", ""),
        "reverse_barrier_eV": summary.get("reverse_barrier_eV", ""),
        "reaction_energy_eV": summary.get("reaction_energy_eV", ""),
        "max_force_eVA": summary.get("max_force_eVA", ""),
        "neb_dir": summary.get("neb_dir", ""),
    }

