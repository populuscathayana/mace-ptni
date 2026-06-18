#!/usr/bin/env python
"""Split a large extxyz file into frame-count chunks without ASE."""

from __future__ import annotations

import argparse
from pathlib import Path


def iter_frame_text(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        frame_index = 0
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            if not natoms_line.strip():
                continue
            try:
                natoms = int(natoms_line.strip())
            except ValueError as exc:
                raise ValueError(f"frame {frame_index}: invalid natoms line {natoms_line!r}") from exc
            comment = handle.readline()
            if not comment:
                raise ValueError(f"frame {frame_index}: missing comment line")
            atom_lines = [handle.readline() for _ in range(natoms)]
            if len(atom_lines) != natoms or any(line == "" for line in atom_lines):
                raise ValueError(f"frame {frame_index}: incomplete atom block")
            yield natoms_line, comment, atom_lines
            frame_index += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk a large extxyz file by number of frames.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--frames-per-chunk", type=int, default=2000)
    args = parser.parse_args()

    if args.frames_per_chunk < 1:
        raise SystemExit("--frames-per-chunk must be >= 1")

    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_path.stem

    chunk_index = -1
    frame_in_chunk = 0
    total_frames = 0
    handle = None
    chunk_paths = []

    try:
        for natoms_line, comment, atom_lines in iter_frame_text(input_path):
            if handle is None or frame_in_chunk >= args.frames_per_chunk:
                if handle is not None:
                    handle.close()
                chunk_index += 1
                frame_in_chunk = 0
                chunk_path = out_dir / f"{prefix}_chunk{chunk_index:04d}.extxyz"
                chunk_paths.append(chunk_path)
                handle = chunk_path.open("w", encoding="utf-8", newline="")

            handle.write(natoms_line)
            handle.write(comment)
            handle.writelines(atom_lines)
            frame_in_chunk += 1
            total_frames += 1
    finally:
        if handle is not None:
            handle.close()

    print(f"Input: {input_path}")
    print(f"Output dir: {out_dir}")
    print(f"Frames per chunk: {args.frames_per_chunk}")
    print(f"Total frames: {total_frames}")
    print(f"Chunks: {len(chunk_paths)}")
    for path in chunk_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
