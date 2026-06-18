#!/usr/bin/env python
"""Create interactive Plotly parity plots for MACE predictions.

The HTML contains energy-per-atom and force-component parity plots. A dropdown
switches between train/valid/test while keeping fixed global x/y ranges.
Point colors encode local 2D point density.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path


KEY_RE = re.compile(r'(\w+)=(".*?"|\S+)')


def parse_info(comment: str) -> dict[str, str]:
    return {key: value.strip('"') for key, value in KEY_RE.findall(comment)}


def parse_properties(comment: str) -> dict[str, tuple[int, int]]:
    info = parse_info(comment)
    prop_text = info.get("Properties")
    if not prop_text:
        raise ValueError(f"comment line has no Properties=: {comment[:200]!r}")
    tokens = prop_text.split(":")
    props = {}
    column = 0
    for i in range(0, len(tokens) - 2, 3):
        name = tokens[i]
        count = int(tokens[i + 2])
        props[name] = (column, count)
        column += count
    return props


def iter_frames(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        frame_index = 0
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            if not natoms_line.strip():
                continue
            natoms = int(natoms_line.strip())
            comment = handle.readline()
            if not comment:
                raise ValueError(f"frame {frame_index}: missing comment line")
            atom_lines = [handle.readline() for _ in range(natoms)]
            if len(atom_lines) != natoms or any(line == "" for line in atom_lines):
                raise ValueError(f"frame {frame_index}: incomplete atom block")
            yield frame_index, natoms, comment.strip(), atom_lines
            frame_index += 1


def choose_info_key(info: dict[str, str], requested: str, ref_key: str) -> str | None:
    if requested in info:
        return requested
    candidates = [
        key for key in info
        if key.lower().endswith("energy")
        and key != ref_key
        and not key.lower().startswith("ref")
    ]
    return sorted(candidates)[0] if candidates else None


def choose_array_key(props: dict[str, tuple[int, int]], requested: str, ref_key: str) -> str | None:
    if requested in props:
        return requested
    candidates = [
        key for key in props
        if key.lower().endswith("forces")
        and key != ref_key
        and not key.lower().startswith("ref")
    ]
    return sorted(candidates)[0] if candidates else None


def parse_vector_components(lines: list[str], props: dict[str, tuple[int, int]], key: str):
    start, count = props[key]
    if count != 3:
        raise ValueError(f"{key} has {count} columns, expected 3")
    for line in lines:
        parts = line.split()
        yield float(parts[start]), float(parts[start + 1]), float(parts[start + 2])


def reservoir_add_pair(sample: list[tuple[float, float]], pair: tuple[float, float], seen: int, limit: int, rng) -> None:
    if limit <= 0:
        return
    if len(sample) < limit:
        sample.append(pair)
    else:
        j = rng.randrange(seen)
        if j < limit:
            sample[j] = pair


def load_split(
    split: str,
    path: Path,
    ref_energy_key: str,
    pred_energy_key: str,
    ref_forces_key: str,
    pred_forces_key: str,
    max_force_points: int,
    seed: int,
) -> dict:
    rng = random.Random(f"{seed}:{split}")
    energy_pairs = []
    force_sample = []
    force_seen = 0
    frames = 0
    atoms = 0

    used_pred_energy_key = None
    used_pred_forces_key = None

    for frame_index, natoms, comment, atom_lines in iter_frames(path):
        info = parse_info(comment)
        props = parse_properties(comment)
        ekey = choose_info_key(info, pred_energy_key, ref_energy_key)
        fkey = choose_array_key(props, pred_forces_key, ref_forces_key)
        if frame_index == 0:
            used_pred_energy_key = ekey
            used_pred_forces_key = fkey

        if ref_energy_key in info and ekey:
            ref_e = float(info[ref_energy_key]) / natoms
            pred_e = float(info[ekey]) / natoms
            energy_pairs.append((ref_e, pred_e))

        if ref_forces_key in props and fkey:
            ref_forces = parse_vector_components(atom_lines, props, ref_forces_key)
            pred_forces = parse_vector_components(atom_lines, props, fkey)
            for ref_vec, pred_vec in zip(ref_forces, pred_forces):
                for ref_value, pred_value in zip(ref_vec, pred_vec):
                    force_seen += 1
                    reservoir_add_pair(force_sample, (ref_value, pred_value), force_seen, max_force_points, rng)

        frames += 1
        atoms += natoms
        if frames % 1000 == 0:
            print(f"{split}: parsed {frames} frames")

    return {
        "split": split,
        "frames": frames,
        "atoms": atoms,
        "energy_pairs": energy_pairs,
        "force_pairs": force_sample,
        "force_components_total": force_seen,
        "force_components_sampled": len(force_sample),
        "pred_energy_key": used_pred_energy_key or "",
        "pred_forces_key": used_pred_forces_key or "",
    }


def finite_min_max(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if math.isfinite(v)]
    return min(values), max(values)


def padded_range(min_value: float, max_value: float, pad_fraction: float = 0.04) -> list[float]:
    if min_value == max_value:
        pad = max(abs(min_value) * 0.05, 1.0)
    else:
        pad = (max_value - min_value) * pad_fraction
    return [min_value - pad, max_value + pad]


def density_colors(pairs: list[tuple[float, float]], axis_range: list[float], bins: int) -> list[float]:
    if not pairs:
        return []
    x0, x1 = axis_range
    y0, y1 = axis_range
    if x1 <= x0 or y1 <= y0:
        return [0.0] * len(pairs)
    counts = {}
    indices = []
    for x, y in pairs:
        ix = int((x - x0) / (x1 - x0) * bins)
        iy = int((y - y0) / (y1 - y0) * bins)
        ix = max(0, min(bins - 1, ix))
        iy = max(0, min(bins - 1, iy))
        key = (ix, iy)
        counts[key] = counts.get(key, 0) + 1
        indices.append(key)
    return [math.log10(counts[key]) for key in indices]


def stats(pairs: list[tuple[float, float]], multiplier: float = 1000.0) -> dict:
    if not pairs:
        return {"mae": None, "rmse": None}
    diffs = [pred - ref for ref, pred in pairs]
    mae = sum(abs(d) for d in diffs) / len(diffs) * multiplier
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs)) * multiplier
    return {"mae": mae, "rmse": rmse}


def load_exact_metrics(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    metrics = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") != "overall":
                continue
            split = row.get("split", "")
            if not split:
                continue
            metrics[split] = {
                "frames": int(row["frames"]),
                "energy_rmse": float(row["energy_rmse_mev_atom"]),
                "force_rmse": float(row["force_component_rmse_mev_A"]),
            }
    return metrics


def build_html(payload: dict) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MACE Parity Density Plots</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #222; background: #fff; }}
    header {{ padding: 16px 20px 10px; border-bottom: 1px solid #ddd; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; font-weight: 650; }}
    .controls {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
    select {{ font-size: 14px; padding: 5px 8px; }}
    .meta {{ font-size: 13px; color: #555; }}
    .wrap {{ display: grid; grid-template-columns: 1fr; gap: 10px; padding: 10px 14px 18px; }}
    .plot {{ height: 520px; border-bottom: 1px solid #eee; }}
    @media (min-width: 1200px) {{
      .wrap {{ grid-template-columns: 1fr 1fr; }}
      .plot {{ height: 700px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>MACE vs DFT Parity Plots</h1>
    <div class="controls">
      <label>Split:
        <select id="splitSelect"></select>
      </label>
      <span id="summary" class="meta"></span>
    </div>
  </header>
  <div class="wrap">
    <div id="energyPlot" class="plot"></div>
    <div id="forcePlot" class="plot"></div>
  </div>
  <script>
    const DATA = {payload_json};
    const select = document.getElementById('splitSelect');
    const summary = document.getElementById('summary');
    DATA.splits.forEach(s => {{
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = s;
      select.appendChild(opt);
    }});

    function traceFor(split, kind) {{
      const d = DATA.data[split][kind];
      return {{
        type: 'scattergl',
        mode: 'markers',
        x: d.x,
        y: d.y,
        marker: {{
          size: kind === 'energy' ? 6 : 3,
          opacity: kind === 'energy' ? 0.72 : 0.45,
          color: d.density,
          colorscale: 'Viridis',
          colorbar: {{
            title: {{ text: 'log10<br>local density' }},
            thickness: 14
          }}
        }},
        hovertemplate:
          kind === 'energy'
          ? 'DFT=%{{x:.8f}} eV/atom<br>MACE=%{{y:.8f}} eV/atom<br>log density=%{{marker.color:.2f}}<extra>' + split + '</extra>'
          : 'DFT=%{{x:.5f}} eV/A<br>MACE=%{{y:.5f}} eV/A<br>log density=%{{marker.color:.2f}}<extra>' + split + '</extra>',
        name: split
      }};
    }}

    function lineTrace(range) {{
      return {{
        type: 'scatter',
        mode: 'lines',
        x: range,
        y: range,
        line: {{ color: '#222', width: 1.2, dash: 'dash' }},
        hoverinfo: 'skip',
        showlegend: false
      }};
    }}

    function layout(kind, split) {{
      const meta = DATA.data[split];
      const d = DATA.data[split][kind];
      const range = DATA.ranges[kind];
      const unit = kind === 'energy' ? 'eV/atom' : 'eV/A';
      const title = kind === 'energy'
        ? `Energy parity (${{split}})`
        : `Force component parity (${{split}})`;
      const sub = kind === 'energy'
        ? `E RMSE ${{(meta.exact_metrics.energy_rmse ?? d.rmse).toFixed(4)}} meV/atom, MAE ${{d.mae.toFixed(4)}} meV/atom`
        : `F RMSE ${{(meta.exact_metrics.force_rmse ?? d.rmse).toFixed(4)}} meV/A, MAE ${{d.mae.toFixed(4)}} meV/A; sampled ${{d.n.toLocaleString()}} / ${{DATA.data[split].force_total.toLocaleString()}} components`;
      return {{
        title: {{ text: title + '<br><sup>' + sub + '</sup>', x: 0.05 }},
        margin: {{ l: 70, r: 70, t: 72, b: 64 }},
        xaxis: {{ title: 'DFT ' + (kind === 'energy' ? 'energy' : 'force component') + ' (' + unit + ')', range, scaleanchor: 'y', scaleratio: 1, zeroline: false }},
        yaxis: {{ title: 'MACE ' + (kind === 'energy' ? 'energy' : 'force component') + ' (' + unit + ')', range, zeroline: false }},
        showlegend: false,
        template: 'plotly_white'
      }};
    }}

    function render(split) {{
      Plotly.react('energyPlot', [traceFor(split, 'energy'), lineTrace(DATA.ranges.energy)], layout('energy', split), {{ responsive: true }});
      Plotly.react('forcePlot', [traceFor(split, 'force'), lineTrace(DATA.ranges.force)], layout('force', split), {{ responsive: true }});
      const meta = DATA.data[split];
      summary.textContent = `${{split}}: frames=${{meta.frames.toLocaleString()}}, atoms=${{meta.atoms.toLocaleString()}}, force sample=${{meta.force.n.toLocaleString()}}/${{meta.force_total.toLocaleString()}}`;
    }}

    select.addEventListener('change', () => render(select.value));
    render(DATA.splits[0]);
  </script>
</body>
</html>
"""


def parse_pred_arg(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use SPLIT=PATH")
    split, path = text.split("=", 1)
    return split.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build interactive MACE parity density HTML.")
    parser.add_argument("--pred", action="append", type=parse_pred_arg, required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    parser.add_argument("--ref-energy-key", default="REF_energy")
    parser.add_argument("--pred-energy-key", default="MACE_energy")
    parser.add_argument("--ref-forces-key", default="REF_forces")
    parser.add_argument("--pred-forces-key", default="MACE_forces")
    parser.add_argument("--metrics-csv", type=Path, help="Optional score_mace_predictions_extxyz.py CSV for exact split RMSE values.")
    parser.add_argument("--max-force-points", type=int, default=200000)
    parser.add_argument("--density-bins", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    loaded = {}
    exact_metrics = load_exact_metrics(args.metrics_csv.resolve() if args.metrics_csv else None)
    for split, path in args.pred:
        print(f"Loading {split}: {path}")
        loaded[split] = load_split(
            split=split,
            path=path.resolve(),
            ref_energy_key=args.ref_energy_key,
            pred_energy_key=args.pred_energy_key,
            ref_forces_key=args.ref_forces_key,
            pred_forces_key=args.pred_forces_key,
            max_force_points=args.max_force_points,
            seed=args.seed,
        )

    all_energy = [v for data in loaded.values() for pair in data["energy_pairs"] for v in pair]
    all_force = [v for data in loaded.values() for pair in data["force_pairs"] for v in pair]
    erange = padded_range(*finite_min_max(all_energy))
    frange = padded_range(*finite_min_max(all_force))

    payload = {
        "splits": list(loaded.keys()),
        "ranges": {"energy": erange, "force": frange},
        "data": {},
    }
    for split, data in loaded.items():
        energy_pairs = data["energy_pairs"]
        force_pairs = data["force_pairs"]
        energy_stats = stats(energy_pairs, multiplier=1000.0)
        force_stats = stats(force_pairs, multiplier=1000.0)
        payload["data"][split] = {
            "frames": data["frames"],
            "atoms": data["atoms"],
            "force_total": data["force_components_total"],
            "exact_metrics": exact_metrics.get(split, {}),
            "energy": {
                "x": [x for x, _ in energy_pairs],
                "y": [y for _, y in energy_pairs],
                "density": density_colors(energy_pairs, erange, args.density_bins),
                "n": len(energy_pairs),
                **energy_stats,
            },
            "force": {
                "x": [x for x, _ in force_pairs],
                "y": [y for _, y in force_pairs],
                "density": density_colors(force_pairs, frange, args.density_bins),
                "n": len(force_pairs),
                **force_stats,
            },
        }

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(build_html(payload), encoding="utf-8")
    print(f"HTML: {args.out_html.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
