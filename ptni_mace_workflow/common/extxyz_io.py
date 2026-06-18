#!/usr/bin/env python
"""Shared extxyz conventions used by the PtNi MACE workflow."""

from __future__ import annotations

REF_ENERGY_KEY = "REF_energy"
REF_FORCES_KEY = "REF_forces"
PRED_ENERGY_KEY = "MACE_energy"
PRED_FORCES_KEY = "MACE_forces"


def split_names() -> tuple[str, str, str]:
    return ("train", "valid", "test")
