#!/usr/bin/env python
"""Export a MACE training checkpoint .pt to an evaluable .model file.

MACE checkpoint .pt files are usually dictionaries containing a model
state_dict plus optimizer/training state. mace_eval_configs expects a serialized
MACE model object. Per MACE maintainer guidance, conversion needs a non-compiled
template .model created with the same architecture/hyperparameters; this script
loads that template model, applies checkpoint["model"], and saves a new .model.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load checkpoint['model'] into a same-hyperparameter MACE .model template."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Training checkpoint .pt")
    parser.add_argument(
        "--template-model",
        type=Path,
        required=True,
        help="Existing non-compiled .model with the same architecture/hyperparameters.",
    )
    parser.add_argument("--output-model", type=Path, required=True, help="Output .model")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require exact state_dict key match. Default allows non-strict load and reports differences.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    checkpoint_path = args.checkpoint.resolve()
    template_path = args.template_model.resolve()
    output_path = args.output_model.resolve()

    if "compiled" in template_path.name.lower():
        raise SystemExit(
            "Do not use a compiled .model as template. Use the ordinary saved .model file."
        )

    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
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

    model = torch.load(template_path, map_location=args.device, weights_only=False)
    if not hasattr(model, "load_state_dict"):
        raise SystemExit(f"Template is not a torch module-like model: {type(model)}")

    result = model.load_state_dict(state, strict=args.strict)
    if not args.strict:
        print(f"Missing keys: {list(result.missing_keys)}")
        print(f"Unexpected keys: {list(result.unexpected_keys)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output_path)

    reloaded = torch.load(output_path, map_location=args.device, weights_only=False)
    if not hasattr(reloaded, "to"):
        raise SystemExit("Saved output does not look evaluable: missing .to method")

    print(f"Wrote evaluable model: {output_path}")
    print("You can now use it with mace_eval_configs --model <output.model>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
