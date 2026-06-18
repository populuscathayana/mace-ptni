#!/usr/bin/env python
"""Inspect MACE run settings from logs and checkpoints.

Use this inside the Python environment where torch/MACE is installed. The log
part works without torch; checkpoint/model inspection imports torch lazily.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


IMPORTANT_KEYS = [
    "name",
    "seed",
    "device",
    "default_dtype",
    "foundation_model",
    "train_file",
    "valid_file",
    "test_file",
    "energy_key",
    "forces_key",
    "E0s",
    "model",
    "r_max",
    "radial_type",
    "num_radial_basis",
    "num_cutoff_basis",
    "distance_transform",
    "interaction",
    "interaction_first",
    "max_ell",
    "correlation",
    "num_interactions",
    "hidden_irreps",
    "MLP_irreps",
    "radial_MLP",
    "num_channels",
    "max_L",
    "gate",
    "scaling",
    "avg_num_neighbors",
    "compute_avg_num_neighbors",
    "loss",
    "energy_weight",
    "forces_weight",
    "config_type_weights",
    "optimizer",
    "batch_size",
    "valid_batch_size",
    "lr",
    "weight_decay",
    "amsgrad",
    "scheduler",
    "lr_factor",
    "scheduler_patience",
    "max_num_epochs",
    "patience",
    "clip_grad",
    "swa",
    "swa_lr",
    "start_swa",
]


def extract_namespace_text(log_text: str) -> str | None:
    match = re.search(r"Configuration: Namespace\((.*)\)", log_text)
    return match.group(1) if match else None


def parse_namespace_items(namespace_body: str) -> dict[str, str]:
    items = {}
    # Split on ", key=" boundaries without trying to fully parse non-literal
    # objects such as KeySpecification(...).
    parts = re.split(r", (?=[A-Za-z_][A-Za-z0-9_]*=)", namespace_body)
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        items[key.strip()] = value.strip()
    return items


def literal_or_text(value: str):
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def print_log_settings(log_path: Path) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    body = extract_namespace_text(text)
    print(f"\n# Log: {log_path}")
    if body is None:
        print("No 'Configuration: Namespace(...)' line found.")
    else:
        parsed = parse_namespace_items(body)
        print("\n## Important command/settings")
        for key in IMPORTANT_KEYS:
            if key in parsed:
                print(f"{key}: {literal_or_text(parsed[key])}")

        extras = sorted(set(parsed) - set(IMPORTANT_KEYS))
        print(f"\n## Other keys in Namespace: {len(extras)}")
        print(", ".join(extras))

    print("\n## Model/optimizer lines from log")
    patterns = [
        "MACE version",
        "Atomic Numbers used",
        "Atomic Energies used",
        "Average number of neighbors",
        "MODEL DETAILS",
        "filtered elements",
        "Message passing",
        "layers",
        "Radial cutoff",
        "Distance transform",
        "OPTIMIZER INFORMATION",
        "Batch size",
        "Number of gradient updates",
        "Learning rate",
        "WeightedEnergyForcesLoss",
        "gradient clipping",
    ]
    for line in text.splitlines():
        if any(pattern in line for pattern in patterns):
            print(line)


def summarize_tensor_dict(state_dict: dict, max_tensors: int) -> None:
    total = 0
    rows = []
    for key, value in state_dict.items():
        if hasattr(value, "numel"):
            n = int(value.numel())
            total += n
            shape = tuple(value.shape)
            dtype = str(value.dtype)
            rows.append((key, n, shape, dtype))
    print(f"Parameter/buffer tensors: {len(rows)}")
    print(f"Total tensor elements: {total}")
    for key, n, shape, dtype in rows[:max_tensors]:
        print(f"  {key}: numel={n}, shape={shape}, dtype={dtype}")
    if len(rows) > max_tensors:
        print(f"  ... {len(rows) - max_tensors} more tensors")


def print_checkpoint_settings(path: Path, max_tensors: int) -> None:
    import torch

    print(f"\n# Checkpoint/model: {path}")
    obj = torch.load(path, map_location="cpu")
    print(f"Top-level type: {type(obj)}")

    if isinstance(obj, dict):
        print(f"Top-level keys: {list(obj.keys())}")
        for key in ["epoch", "model_config", "config", "args", "atomic_numbers", "r_max"]:
            if key in obj:
                value = obj[key]
                try:
                    print(f"{key}: {json.dumps(value, default=str, indent=2)[:4000]}")
                except Exception:
                    print(f"{key}: {value}")

        state = None
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                print(f"\n## Tensor summary from key '{key}'")
                summarize_tensor_dict(state, max_tensors=max_tensors)
                break
        if state is None:
            print("No tensor state dict found under common keys.")
    else:
        print("Object is not a dict. Trying state_dict() if available.")
        if hasattr(obj, "state_dict"):
            summarize_tensor_dict(obj.state_dict(), max_tensors=max_tensors)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect MACE run settings.")
    parser.add_argument("--log", type=Path, default=None, help="MACE debug or normal log path.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="MACE .pt or .model path.")
    parser.add_argument("--max-tensors", type=int, default=80)
    args = parser.parse_args()

    if args.log is None and args.checkpoint is None:
        raise SystemExit("Provide --log and/or --checkpoint")
    if args.log is not None:
        print_log_settings(args.log.resolve())
    if args.checkpoint is not None:
        print_checkpoint_settings(args.checkpoint.resolve(), args.max_tensors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
