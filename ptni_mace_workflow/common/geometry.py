#!/usr/bin/env python
"""Geometry labels and defaults shared by PtNi benchmark scripts."""

from __future__ import annotations

PT_FCC_SCAN = "3.75:4.15:0.01"
NI_FCC_SCAN = "3.35:3.75:0.01"
DEFAULT_STRAIN_SCAN = "-3:3:1"


def truthy(value: str | int | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
