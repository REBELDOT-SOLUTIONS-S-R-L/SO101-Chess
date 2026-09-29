#!/usr/bin/env python3
"""Validate the structural invariants of an annotated or generated chess HDF5 dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import h5py


PIECE_TYPES = {"pawn", "rook", "knight", "bishop", "queen", "king"}
REQUIRED_TIMESERIES = (
    "actions/joints",
    "actions/pose",
    "obs/articulations/robot/joint_position",
    "obs/eef_pose/chess",
    "obs/object_pose/active_piece",
    "obs/object_pose/destination_square",
)
REQUIRED_CAMERAS = (
    "obs/cameras/top_camera",
    "obs/cameras/right_wrist_camera",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-per-piece", type=int)
    parser.add_argument(
        "--expected-success",
        choices=("true", "false", "any"),
        default="any",
    )
    parser.add_argument("--require-balanced-types", action="store_true")
    parser.add_argument("--require-cameras", action="store_true")
    return parser.parse_args()


def episode_number(name: str) -> int:
    prefix, separator, suffix = name.rpartition("_")
    if not separator or not prefix or not suffix.isdigit():
        raise RuntimeError(f"Unexpected episode group name: {name!r}")
    return int(suffix)


def main() -> int:
    args = parse_args()
    path = args.dataset.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")

    piece_counts: Counter[str] = Counter()
    total_samples = 0
    with h5py.File(path, "r") as dataset:
        if "data" not in dataset or not isinstance(dataset["data"], h5py.Group):
            raise RuntimeError(f"Missing /data group in {path}")
        data = dataset["data"]
        names = sorted(data.keys(), key=episode_number)
        episode_count = len(names)

        if args.expected_episodes is not None and episode_count != args.expected_episodes:
            raise RuntimeError(
                f"Expected {args.expected_episodes} episodes, found {episode_count} in {path}"
            )
        if names and [episode_number(name) for name in names] != list(range(episode_count)):
            raise RuntimeError("Episode groups are not contiguous from demo_0")

        for name in names:
            episode = data[name]
            sample_count = int(episode.attrs.get("num_samples", -1))
            if sample_count <= 0:
                raise RuntimeError(f"{name}: invalid num_samples={sample_count}")
            success = bool(episode.attrs.get("success", False))
            if args.expected_success == "true" and not success:
                raise RuntimeError(f"{name}: expected success=True")
            if args.expected_success == "false" and success:
                raise RuntimeError(f"{name}: expected success=False")

            piece_type = str(episode.attrs.get("piece_type", "")).lower()
            if piece_type not in PIECE_TYPES:
                raise RuntimeError(f"{name}: invalid or missing piece_type={piece_type!r}")
            piece_counts[piece_type] += 1

            for dataset_path in REQUIRED_TIMESERIES:
                if dataset_path not in episode:
                    raise RuntimeError(f"{name}: missing {dataset_path}")
                if episode[dataset_path].shape[0] != sample_count:
                    raise RuntimeError(
                        f"{name}: {dataset_path} has {episode[dataset_path].shape[0]} rows, "
                        f"expected {sample_count}"
                    )
            if args.require_cameras:
                for camera_path in REQUIRED_CAMERAS:
                    if camera_path not in episode:
                        raise RuntimeError(f"{name}: missing {camera_path}")
                    camera = episode[camera_path]
                    if camera.shape != (sample_count, 480, 640, 3):
                        raise RuntimeError(
                            f"{name}: {camera_path} has unexpected shape {camera.shape}"
                        )
            total_samples += sample_count

        recorded_episodes = data.attrs.get("num_episodes")
        if recorded_episodes is not None and int(recorded_episodes) != episode_count:
            raise RuntimeError(
                f"/data num_episodes={recorded_episodes}, but contains {episode_count} groups"
            )
        for attr in ("total", "total_samples"):
            value = data.attrs.get(attr)
            if value is not None and int(value) != total_samples:
                raise RuntimeError(
                    f"/data {attr}={value}, but episode samples sum to {total_samples}"
                )

    if args.expected_per_piece is not None:
        expected = {piece: args.expected_per_piece for piece in PIECE_TYPES}
        if dict(piece_counts) != expected:
            raise RuntimeError(
                f"Expected {args.expected_per_piece} episodes per piece; found {dict(piece_counts)}"
            )
    if args.require_balanced_types:
        if set(piece_counts) != PIECE_TYPES:
            raise RuntimeError(f"Expected all six piece types; found {dict(piece_counts)}")
        if max(piece_counts.values()) - min(piece_counts.values()) > 1:
            raise RuntimeError(f"Piece distribution is not balanced: {dict(piece_counts)}")

    ordered_counts = {piece: piece_counts[piece] for piece in sorted(piece_counts)}
    print(
        f"Validated {path}: episodes={sum(piece_counts.values())}, "
        f"samples={total_samples}, pieces={ordered_counts}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
