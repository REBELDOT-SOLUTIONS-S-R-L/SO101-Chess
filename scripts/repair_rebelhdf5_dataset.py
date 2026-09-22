#!/usr/bin/env python3
"""Repair existing SO-101 chess datasets for RebelHDF5 visualization.

The repair is intentionally limited to two task-specific schema changes:

* replace all physical chess-piece entries in each episode's initial state
  with the one on-board piece, named ``active_piece``;
* remove inactive pieces from legacy per-step scene states as well;
* add ``reference_demo_indices/left`` as an alias of the canonical
  ``reference_demo_indices/chess`` dataset.

Use ``--output`` to preserve the source file or ``--in-place`` to update it
without copying large camera datasets.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import h5py


ACTIVE_PIECE_NAME = "active_piece"
CANONICAL_REFERENCE_KEY = "chess"
REBELHDF5_REFERENCE_KEY = "left"
INITIAL_OBJECT_PATHS = (
    "initial_state/rigid_objects",
    "initial_state/rigid_object",
    "initial_state/objects",
)
STATE_OBJECT_PATHS = (
    "states/rigid_objects",
    "states/rigid_object",
    "states/objects",
)


def _initial_xy(object_group: h5py.Group) -> tuple[float, float]:
    pose = object_group.get("initial_pose")
    if not isinstance(pose, h5py.Dataset):
        pose = object_group.get("root_pose")
    if not isinstance(pose, h5py.Dataset):
        raise ValueError(f"{object_group.name} has no initial_pose or root_pose dataset")

    if pose.ndim >= 2 and pose.shape[-2:] == (4, 4):
        matrix = pose[0] if pose.ndim == 3 else pose[...]
        return float(matrix[0, 3]), float(matrix[1, 3])

    values = pose[...].reshape(-1)
    if len(values) < 2:
        raise ValueError(f"Unsupported initial_pose shape at {pose.name}: {pose.shape}")
    return float(values[0]), float(values[1])


def _find_active_piece(rigid_objects: h5py.Group) -> str:
    if ACTIVE_PIECE_NAME in rigid_objects:
        return ACTIVE_PIECE_NAME

    candidates: list[tuple[float, str]] = []
    for object_name, object_group in rigid_objects.items():
        if not isinstance(object_group, h5py.Group):
            continue
        x, y = _initial_xy(object_group)
        candidates.append((math.hypot(x, y), object_name))

    if not candidates:
        raise ValueError(f"No rigid object initial poses found under {rigid_objects.name}")

    candidates.sort()
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 1.0:
        details = ", ".join(f"{name}={distance:.3f}" for distance, name in candidates)
        raise ValueError(
            f"Could not uniquely identify the on-board active piece in {rigid_objects.name}: {details}"
        )
    return candidates[0][1]


def _normalize_object_group(object_group: h5py.Group, active_name: str) -> None:
    """Keep one object in a standard or legacy object-state group."""
    resolved_active_name = active_name
    if resolved_active_name not in object_group:
        if ACTIVE_PIECE_NAME in object_group:
            resolved_active_name = ACTIVE_PIECE_NAME
        else:
            # This covers a partially repaired legacy episode whose initial
            # state is already aliased but whose per-step states are not.
            resolved_active_name = _find_active_piece(object_group)

    for object_name in list(object_group.keys()):
        if object_name != resolved_active_name:
            del object_group[object_name]
    if resolved_active_name != ACTIVE_PIECE_NAME:
        object_group.move(resolved_active_name, ACTIVE_PIECE_NAME)


def repair_file(path: Path) -> tuple[int, int]:
    repaired_initial_states = 0
    added_reference_aliases = 0

    with h5py.File(path, "r+") as dataset:
        data_group = dataset.get("data")
        if not isinstance(data_group, h5py.Group):
            raise ValueError(f"{path} does not contain a /data group")

        # Validate every active-piece choice before deleting anything. This
        # keeps an ambiguous episode from leaving an in-place repair half done.
        active_piece_by_demo: dict[str, tuple[str, str]] = {}
        for demo_name, demo_group in data_group.items():
            if not isinstance(demo_group, h5py.Group):
                continue
            for object_path in INITIAL_OBJECT_PATHS:
                rigid_objects = demo_group.get(object_path)
                if isinstance(rigid_objects, h5py.Group) and len(rigid_objects) > 0:
                    active_piece_by_demo[demo_name] = (
                        object_path,
                        _find_active_piece(rigid_objects),
                    )
                    break

        for demo_name, demo_group in data_group.items():
            if not isinstance(demo_group, h5py.Group):
                continue

            if demo_name in active_piece_by_demo:
                object_path, active_name = active_piece_by_demo[demo_name]
                rigid_objects = demo_group[object_path]
                _normalize_object_group(rigid_objects, active_name)
                repaired_initial_states += 1

                for state_path in STATE_OBJECT_PATHS:
                    state_objects = demo_group.get(state_path)
                    if isinstance(state_objects, h5py.Group) and len(state_objects) > 0:
                        _normalize_object_group(state_objects, active_name)

            references = demo_group.get("reference_demo_indices")
            if (
                isinstance(references, h5py.Group)
                and CANONICAL_REFERENCE_KEY in references
                and REBELHDF5_REFERENCE_KEY not in references
            ):
                # A hard link adds the schema alias without duplicating data.
                references[REBELHDF5_REFERENCE_KEY] = references[CANONICAL_REFERENCE_KEY]
                added_reference_aliases += 1

    return repaired_initial_states, added_reference_aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Existing HDF5 dataset to repair.")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--output",
        type=Path,
        help="Write a repaired copy here, preserving the input file.",
    )
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="Modify the input file directly (recommended for large camera datasets).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.dataset.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    target = source
    if args.output is not None:
        target = args.output.expanduser().resolve()
        if target == source:
            raise ValueError("--output must differ from the input; use --in-place instead")
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    initial_state_count, reference_count = repair_file(target)
    print(
        f"Repaired {target}: normalized {initial_state_count} initial states; "
        f"added {reference_count} source-reference aliases."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
