"""Add the virtual destination square to existing chess Mimic initial states.

The stored destination pose is robot-root-relative; initial-state rigid-object
poses are relative to the environment origin. This script converts between
those frames and is safe to rerun on files that were already updated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"Invalid robot root quaternion: {quaternion}")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def destination_initial_pose(demo: h5py.Group) -> np.ndarray:
    root_pose = demo["initial_state/articulations/robot/root_pose"][0]
    destination_poses = demo["obs/object_pose/destination_square"]
    destination_pose_b = destination_poses[0]
    if not np.allclose(destination_poses[...], destination_pose_b, atol=1e-5, rtol=0):
        raise ValueError(f"Destination pose changes during {demo.name}; cannot infer a single initial pose")

    robot_pose_env = np.eye(4, dtype=destination_pose_b.dtype)
    robot_pose_env[:3, :3] = quaternion_wxyz_to_matrix(root_pose[3:7])
    robot_pose_env[:3, 3] = root_pose[:3]
    return robot_pose_env @ destination_pose_b


def backfill_file(path: Path, dry_run: bool) -> tuple[int, int]:
    added = 0
    already_present = 0
    with h5py.File(path, "r" if dry_run else "r+") as dataset:
        for demo_name in dataset["data"]:
            demo = dataset["data"][demo_name]
            rigid_objects = demo["initial_state/rigid_objects"]
            if "destination_square" in rigid_objects:
                already_present += 1
                continue

            pose = destination_initial_pose(demo)
            if not dry_run:
                destination = rigid_objects.create_group("destination_square")
                destination.create_dataset("initial_pose", data=pose[None], compression="gzip")
                destination.create_dataset(
                    "scale", data=np.ones((1, 3), dtype=pose.dtype), compression="gzip"
                )
            added += 1
    return added, already_present


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="Generated chess HDF5 files to update in place.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count without changing files.")
    args = parser.parse_args()

    for path in args.files:
        added, already_present = backfill_file(path, args.dry_run)
        verb = "would add" if args.dry_run else "added"
        print(f"{path}: {verb} {added}, already present {already_present}")


if __name__ == "__main__":
    main()
