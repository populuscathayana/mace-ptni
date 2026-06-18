#!/usr/bin/env python
"""Check MACE training progress from log and JSON-lines result files.

The script parses validation metrics from a MACE .log file, summarizes best
epochs, epoch timing, spikes/plateaus, available checkpoints, and optionally
plots validation curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


VAL_RE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*?"
    r"(?:(?P<label>Initial)|Epoch (?P<epoch>\d+)): head: (?P<head>\S+), "
    r"loss=(?P<loss>[-+0-9.eE]+), "
    r"RMSE_E_per_atom=\s*(?P<rmse_e>[-+0-9.eE]+) meV, "
    r"RMSE_F=\s*(?P<rmse_f>[-+0-9.eE]+) meV / A"
)

DATASET_RE = re.compile(
    r"Total number of configurations: train=(?P<train>\d+), valid=(?P<valid>\d+), tests=\[(?P<tests>.*?)\]"
)
NEIGH_RE = re.compile(r"Average number of neighbors: (?P<value>[-+0-9.eE]+)")
MODEL_LINES = [
    "MACE version",
    "Using foundation model",
    "Atomic Numbers used",
    "Atomic Energies used",
    "Average number of neighbors",
    "Message passing",
    "layers, each with correlation",
    "Radial cutoff",
    "Distance transform",
    "Batch size",
    "Number of gradient updates",
    "Learning rate",
    "WeightedEnergyForcesLoss",
    "gradient clipping",
]


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f")


def parse_log(log_path: Path) -> tuple[list[dict], dict]:
    rows = []
    metadata = {
        "model_lines": [],
        "dataset_line": "",
        "train_configs": "",
        "valid_configs": "",
        "test_configs": "",
        "avg_neighbors": "",
    }
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = VAL_RE.search(line)
        if match:
            epoch_text = match.group("epoch")
            epoch = -1 if epoch_text is None else int(epoch_text)
            label = "Initial" if epoch == -1 else f"Epoch {epoch}"
            rows.append(
                {
                    "timestamp": match.group("time"),
                    "datetime": parse_time(match.group("time")),
                    "label": label,
                    "epoch": epoch,
                    "head": match.group("head"),
                    "loss": float(match.group("loss")),
                    "rmse_e_per_atom_mev": float(match.group("rmse_e")),
                    "rmse_f_mev_per_a": float(match.group("rmse_f")),
                }
            )

        dataset_match = DATASET_RE.search(line)
        if dataset_match:
            metadata["dataset_line"] = line
            metadata["train_configs"] = int(dataset_match.group("train"))
            metadata["valid_configs"] = int(dataset_match.group("valid"))
            metadata["test_configs"] = dataset_match.group("tests")

        neigh_match = NEIGH_RE.search(line)
        if neigh_match:
            metadata["avg_neighbors"] = float(neigh_match.group("value"))

        if any(pattern in line for pattern in MODEL_LINES):
            metadata["model_lines"].append(line)

    rows.sort(key=lambda row: row["epoch"])
    return rows, metadata


def parse_json_results(path: Path | None) -> dict:
    summary = {
        "opt_steps_by_epoch": Counter(),
        "opt_loss_by_epoch": defaultdict(list),
        "eval_rows": [],
        "last_mode": "",
        "last_epoch": "",
    }
    if path is None or not path.is_file():
        return summary
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            mode = item.get("mode", "")
            epoch = item.get("epoch")
            summary["last_mode"] = mode
            summary["last_epoch"] = epoch
            if mode == "opt" and epoch is not None:
                summary["opt_steps_by_epoch"][epoch] += 1
                if "loss" in item:
                    summary["opt_loss_by_epoch"][epoch].append(float(item["loss"]))
            elif mode == "eval":
                summary["eval_rows"].append(item)
    return summary


def safe_mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def add_derived_metrics(rows: list[dict]) -> None:
    prev = None
    best_loss = math.inf
    best_e = math.inf
    best_f = math.inf
    for row in rows:
        if prev is None:
            row["minutes_since_previous_validation"] = ""
        else:
            delta = row["datetime"] - prev["datetime"]
            row["minutes_since_previous_validation"] = delta.total_seconds() / 60.0

        if row["epoch"] >= 0:
            best_loss = min(best_loss, row["loss"])
            best_e = min(best_e, row["rmse_e_per_atom_mev"])
            best_f = min(best_f, row["rmse_f_mev_per_a"])
            row["loss_vs_best_ratio"] = row["loss"] / best_loss if best_loss else 1.0
            row["force_vs_best_ratio"] = row["rmse_f_mev_per_a"] / best_f if best_f else 1.0
            row["energy_vs_best_ratio"] = row["rmse_e_per_atom_mev"] / best_e if best_e else 1.0
        else:
            row["loss_vs_best_ratio"] = ""
            row["force_vs_best_ratio"] = ""
            row["energy_vs_best_ratio"] = ""
        prev = row


def best_row(rows: list[dict], key: str) -> dict | None:
    epoch_rows = [row for row in rows if row["epoch"] >= 0]
    if not epoch_rows:
        return None
    return min(epoch_rows, key=lambda row: row[key])


def format_row(row: dict | None, key: str) -> str:
    if row is None:
        return "n/a"
    value = row[key]
    return f"epoch {row['epoch']} ({value:.6g})"


def checkpoint_epochs(checkpoints_dir: Path | None) -> list[int]:
    if checkpoints_dir is None or not checkpoints_dir.is_dir():
        return []
    epochs = []
    for path in checkpoints_dir.glob("*.pt"):
        match = re.search(r"epoch-(\d+)", path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(set(epochs))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "timestamp",
        "label",
        "epoch",
        "head",
        "loss",
        "rmse_e_per_atom_mev",
        "rmse_f_mev_per_a",
        "minutes_since_previous_validation",
        "loss_vs_best_ratio",
        "energy_vs_best_ratio",
        "force_vs_best_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(
    path: Path,
    log_path: Path,
    rows: list[dict],
    metadata: dict,
    json_summary: dict,
    ckpt_epochs: list[int],
    max_epochs: int | None,
) -> None:
    epoch_rows = [row for row in rows if row["epoch"] >= 0]
    initial = next((row for row in rows if row["epoch"] == -1), None)
    latest = epoch_rows[-1] if epoch_rows else None
    best_loss = best_row(rows, "loss")
    best_energy = best_row(rows, "rmse_e_per_atom_mev")
    best_force = best_row(rows, "rmse_f_mev_per_a")
    durations = [
        row["minutes_since_previous_validation"]
        for row in epoch_rows
        if isinstance(row.get("minutes_since_previous_validation"), float)
    ]
    avg_minutes = safe_mean(durations)

    warnings = []
    if latest and best_loss and latest["epoch"] > best_loss["epoch"]:
        warnings.append(
            f"Latest validation loss is not the best; best loss was epoch {best_loss['epoch']}."
        )
    if latest and best_force and latest["rmse_f_mev_per_a"] > best_force["rmse_f_mev_per_a"] * 1.25:
        warnings.append(
            f"Latest force RMSE is >25% above best force RMSE; watch for LR oscillation or overfitting."
        )
    if latest and best_energy and latest["rmse_e_per_atom_mev"] > best_energy["rmse_e_per_atom_mev"] * 1.5:
        warnings.append(
            f"Latest energy RMSE is >50% above best energy RMSE; energy validation is oscillating."
        )
    if not warnings:
        warnings.append("No hard stop signal yet; continue unless validation keeps worsening for several epochs.")

    remaining_text = "unknown"
    if max_epochs is not None and latest is not None and isinstance(avg_minutes, float):
        remaining = max(max_epochs - latest["epoch"] - 1, 0)
        remaining_text = f"about {remaining * avg_minutes / 60.0:.1f} hours for {remaining} more epochs"

    lines = []
    lines.append("# MACE Training Progress Report")
    lines.append("")
    lines.append(f"- Log: `{log_path}`")
    lines.append(f"- Parsed validation rows: {len(rows)}")
    if metadata.get("train_configs"):
        lines.append(f"- Dataset: train={metadata['train_configs']}, valid={metadata['valid_configs']}, tests={metadata['test_configs']}")
    if metadata.get("avg_neighbors") != "":
        lines.append(f"- Average neighbors: {metadata['avg_neighbors']}")
    if initial:
        lines.append(
            f"- Initial: loss={initial['loss']:.6g}, E={initial['rmse_e_per_atom_mev']:.2f} meV/atom, F={initial['rmse_f_mev_per_a']:.2f} meV/A"
        )
    if latest:
        lines.append(
            f"- Latest epoch {latest['epoch']}: loss={latest['loss']:.6g}, E={latest['rmse_e_per_atom_mev']:.2f} meV/atom, F={latest['rmse_f_mev_per_a']:.2f} meV/A"
        )
    lines.append(f"- Best validation loss: {format_row(best_loss, 'loss')}")
    lines.append(f"- Best validation energy RMSE: {format_row(best_energy, 'rmse_e_per_atom_mev')} meV/atom")
    lines.append(f"- Best validation force RMSE: {format_row(best_force, 'rmse_f_mev_per_a')} meV/A")
    if isinstance(avg_minutes, float):
        lines.append(f"- Average minutes per validated epoch: {avg_minutes:.1f}")
    lines.append(f"- Estimated remaining time: {remaining_text}")
    if ckpt_epochs:
        lines.append(f"- Available checkpoint epochs: {ckpt_epochs}")
    if json_summary["last_mode"]:
        lines.append(f"- Latest JSONL mode/epoch: {json_summary['last_mode']} / {json_summary['last_epoch']}")
    if json_summary["opt_steps_by_epoch"]:
        last_epoch = max(json_summary["opt_steps_by_epoch"])
        losses = json_summary["opt_loss_by_epoch"][last_epoch]
        lines.append(
            f"- Current optimization epoch {last_epoch}: {json_summary['opt_steps_by_epoch'][last_epoch]} opt steps logged, mean opt loss={safe_mean(losses):.6g}"
        )

    lines.append("")
    lines.append("## Checks")
    for warning in warnings:
        lines.append(f"- {warning}")

    lines.append("")
    lines.append("## Model/Optimizer Lines")
    for line in metadata.get("model_lines", []):
        lines.append(f"- {line}")

    lines.append("")
    lines.append("## Validation Table")
    lines.append("")
    lines.append("| epoch | loss | E RMSE meV/atom | F RMSE meV/A | minutes |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in rows:
        epoch = "Initial" if row["epoch"] == -1 else str(row["epoch"])
        minutes = row.get("minutes_since_previous_validation", "")
        minutes_text = "" if minutes == "" else f"{minutes:.1f}"
        lines.append(
            f"| {epoch} | {row['loss']:.8f} | {row['rmse_e_per_atom_mev']:.2f} | {row['rmse_f_mev_per_a']:.2f} | {minutes_text} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_plot(path: Path, rows: list[dict]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    epoch_rows = [row for row in rows if row["epoch"] >= 0]
    if not epoch_rows:
        return False

    epochs = [row["epoch"] for row in epoch_rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(epochs, [row["loss"] for row in epoch_rows], marker="o")
    axes[0].set_ylabel("loss")
    axes[0].set_yscale("log")
    axes[1].plot(epochs, [row["rmse_e_per_atom_mev"] for row in epoch_rows], marker="o", color="tab:orange")
    axes[1].set_ylabel("E RMSE\nmeV/atom")
    axes[2].plot(epochs, [row["rmse_f_mev_per_a"] for row in epoch_rows], marker="o", color="tab:green")
    axes[2].set_ylabel("F RMSE\nmeV/A")
    axes[2].set_xlabel("epoch")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MACE training progress.")
    parser.add_argument("--log", type=Path, required=True, help="MACE .log file.")
    parser.add_argument("--jsonl", type=Path, default=None, help="MACE results *_train.txt JSON-lines file.")
    parser.add_argument("--checkpoints-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("mace_workspace/tmp/training_checks"))
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    log_path = args.log.resolve()
    rows, metadata = parse_log(log_path)
    if not rows:
        raise SystemExit(f"No validation metrics parsed from {log_path}")
    add_derived_metrics(rows)

    out_dir = args.out_dir.resolve()
    stem = log_path.stem
    csv_path = out_dir / f"{stem}_validation_metrics.csv"
    report_path = out_dir / f"{stem}_training_report.md"
    plot_path = out_dir / f"{stem}_curves.png"

    json_summary = parse_json_results(args.jsonl.resolve() if args.jsonl else None)
    ckpt_epochs = checkpoint_epochs(args.checkpoints_dir.resolve() if args.checkpoints_dir else None)

    write_csv(csv_path, rows)
    write_report(report_path, log_path, rows, metadata, json_summary, ckpt_epochs, args.max_epochs)
    plotted = maybe_plot(plot_path, rows) if args.plot else False

    latest = [row for row in rows if row["epoch"] >= 0][-1]
    best_loss = best_row(rows, "loss")
    best_energy = best_row(rows, "rmse_e_per_atom_mev")
    best_force = best_row(rows, "rmse_f_mev_per_a")
    print(f"Parsed validation rows: {len(rows)}")
    print(f"Latest epoch: {latest['epoch']} loss={latest['loss']:.8f} E={latest['rmse_e_per_atom_mev']:.2f} meV/atom F={latest['rmse_f_mev_per_a']:.2f} meV/A")
    print(f"Best loss: epoch {best_loss['epoch']} loss={best_loss['loss']:.8f}")
    print(f"Best energy: epoch {best_energy['epoch']} E={best_energy['rmse_e_per_atom_mev']:.2f} meV/atom")
    print(f"Best force: epoch {best_force['epoch']} F={best_force['rmse_f_mev_per_a']:.2f} meV/A")
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    if args.plot:
        print(f"Plot: {plot_path if plotted else 'not created; matplotlib unavailable'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
