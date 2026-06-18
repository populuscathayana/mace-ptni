#!/usr/bin/env python
"""Export the best MACE checkpoint from a training log to an evaluable .model.

This is useful when MACE finishes and writes a final .model from the last epoch,
but the validation-best checkpoint is an earlier .pt file. The final .model can
serve as the same-architecture template; the script loads the selected
checkpoint state_dict into that template and saves a new .model.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


VAL_RE = re.compile(
    r"Epoch (?P<epoch>\d+): head: \S+, "
    r"loss=(?P<loss>[-+0-9.eE]+), "
    r"RMSE_E_per_atom=\s*(?P<energy>[-+0-9.eE]+) meV, "
    r"RMSE_F=\s*(?P<force>[-+0-9.eE]+) meV / A"
)


def parse_best_epoch(log_path: Path, metric: str) -> tuple[int, float]:
    key_map = {
        "loss": "loss",
        "energy": "energy",
        "force": "force",
    }
    key = key_map[metric]
    best_epoch = None
    best_value = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = VAL_RE.search(line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        value = float(match.group(key))
        if best_value is None or value < best_value:
            best_epoch = epoch
            best_value = value
    if best_epoch is None or best_value is None:
        raise SystemExit(f"No epoch validation rows found in {log_path}")
    return best_epoch, best_value


def find_checkpoint(checkpoints_dir: Path, run_name: str, epoch: int) -> Path:
    patterns = [
        f"{run_name}_run-*_epoch-{epoch}.pt",
        f"*{run_name}*epoch-{epoch}.pt",
        f"*epoch-{epoch}.pt",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(sorted(checkpoints_dir.glob(pattern)))
        if matches:
            break
    if not matches:
        raise SystemExit(
            f"Could not find checkpoint for epoch {epoch} under {checkpoints_dir}. "
            "If it was not saved, rerun with SAVE_ALL_CHECKPOINTS=1 or choose another metric/epoch."
        )
    if len(matches) > 1:
        print(f"Multiple checkpoint matches for epoch {epoch}; using newest:")
        for path in matches:
            print(f"  {path}")
        matches = sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0].resolve()


def export_model(checkpoint_path: Path, template_model_path: Path, output_model_path: Path, strict: bool) -> None:
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"Checkpoint is not a dict: {type(ckpt)}")

    state = None
    for key in ("model", "state_dict", "model_state_dict"):
        if key in ckpt:
            state = ckpt[key]
            print(f"Using checkpoint state key: {key}")
            break
    if state is None:
        raise SystemExit(f"No model state_dict found. Checkpoint keys: {list(ckpt.keys())}")

    model = torch.load(template_model_path, map_location="cpu", weights_only=False)
    if not hasattr(model, "load_state_dict"):
        raise SystemExit(f"Template model is not torch-module-like: {type(model)}")

    result = model.load_state_dict(state, strict=strict)
    if not strict:
        print(f"Missing keys: {list(result.missing_keys)}")
        print(f"Unexpected keys: {list(result.unexpected_keys)}")

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output_model_path)
    print(f"Wrote: {output_model_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export validation-best MACE checkpoint to .model.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--template-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, default=None)
    parser.add_argument("--run-name", default="ptni_binary_mace_ft")
    parser.add_argument(
        "--metric",
        choices=["loss", "energy", "force"],
        default="loss",
        help="Validation metric used to select the checkpoint. Default: loss.",
    )
    parser.add_argument("--epoch", type=int, default=None, help="Manually export this epoch instead of parsing best.")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    log_path = args.log.resolve()
    checkpoints_dir = args.checkpoints_dir.resolve()
    template_model_path = args.template_model.resolve()

    if args.epoch is None:
        epoch, value = parse_best_epoch(log_path, args.metric)
        print(f"Best validation {args.metric}: epoch {epoch}, value {value}")
    else:
        epoch = args.epoch
        print(f"Manual epoch selected: {epoch}")

    checkpoint_path = find_checkpoint(checkpoints_dir, args.run_name, epoch)
    output_model_path = (
        args.output_model.resolve()
        if args.output_model is not None
        else checkpoints_dir / f"{args.run_name}_best_{args.metric}_epoch-{epoch}.model"
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Template model: {template_model_path}")
    print(f"Output model: {output_model_path}")
    export_model(checkpoint_path, template_model_path, output_model_path, args.strict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
