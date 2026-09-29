#!/usr/bin/env python3
"""Re-record selected annotated source demos and fill their original HDF5 IDs.

The target dataset is expected to have the requested ``demo_N`` groups removed.
An untouched template/backup supplies the exact per-type Sobol indices and is
also used to validate piece type, schema, source square, and destination square.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDER = REPO_ROOT / "scripts/record_annotated_demos_balanced.py"
DEFAULT_TEMPLATE = REPO_ROOT / "datasets/annotated_dataset_balanced_lowpoly.before_targeted_rerecord_20260928.hdf5"
DEFAULT_TARGET = REPO_ROOT / "datasets/annotated_dataset_balanced_lowpoly.hdf5"
DEFAULT_REPLACEMENTS = REPO_ROOT / "datasets/annotated_dataset_balanced_lowpoly_targeted_replacements.hdf5"
DEFAULT_IDS = (1, 14, 32, 50)
POSE_PATHS = (
    "initial_state/rigid_objects/active_piece/initial_pose",
    "initial_state/rigid_objects/destination_square/initial_pose",
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Unknown arguments are forwarded to record_annotated_demos_balanced.py.\n"
            "Example: --teleop_device so101leader --port /dev/ttyACM0 --enable_pinocchio"
        ),
    )
    parser.add_argument("--template_dataset", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--target_dataset", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--replacement_dataset", type=Path, default=DEFAULT_REPLACEMENTS)
    parser.add_argument("--episode_ids", type=int, nargs="+", default=list(DEFAULT_IDS))
    parser.add_argument("--task", default="LeIsaac-SO101-Chess-v0-Mimic")
    parser.add_argument(
        "--use_existing_replacements",
        action="store_true",
        help="Validate and merge an already-recorded replacement dataset.",
    )
    args, recorder_args = parser.parse_known_args()
    forbidden = ("--dataset_file", "--num_demos", "--piece_type", "--sobol_plan", "--sobol_start_index")
    conflicts = [
        value
        for value in recorder_args
        if any(value == option or value.startswith(f"{option}=") for option in forbidden)
    ]
    if conflicts:
        parser.error("these recorder options are managed by this workflow: " + ", ".join(conflicts))
    return args, recorder_args


def demo_number(name: str) -> int:
    prefix, separator, suffix = name.rpartition("_")
    if prefix != "demo" or separator != "_" or not suffix.isdigit():
        raise ValueError(f"Unexpected episode group name: {name!r}")
    return int(suffix)


def piece_type(episode: h5py.Group) -> str:
    value = episode.attrs.get("piece_type", "")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip().lower()


def dataset_paths(group: h5py.Group) -> set[str]:
    paths: set[str] = set()

    def collect(name: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset):
            paths.add(name)

    group.visititems(collect)
    return paths


def episode_translation(episode: h5py.Group, path: str) -> np.ndarray:
    pose = np.asarray(episode[path][()])
    if pose.shape != (1, 4, 4):
        raise ValueError(f"{episode.name}/{path} has shape {pose.shape}; expected (1, 4, 4)")
    return pose[0, :3, 3]


def build_plan(template: Path, episode_ids: list[int]) -> tuple[list[str], str]:
    requested = {f"demo_{episode_id}" for episode_id in episode_ids}
    with h5py.File(template, "r") as dataset:
        data = dataset["data"]
        names = sorted(data.keys(), key=demo_number)
        missing = requested - set(names)
        if missing:
            raise ValueError(f"Template is missing requested demos: {sorted(missing, key=demo_number)}")
        ordinal_by_name: dict[str, int] = {}
        seen_by_type: dict[str, int] = {}
        for name in names:
            current_type = piece_type(data[name])
            ordinal_by_name[name] = seen_by_type.get(current_type, 0)
            seen_by_type[current_type] = ordinal_by_name[name] + 1

        ordered_names = [f"demo_{episode_id}" for episode_id in episode_ids]
        entries = [f"{piece_type(data[name])}:{ordinal_by_name[name]}" for name in ordered_names]
    return ordered_names, ",".join(entries)


def records_cameras(dataset: Path) -> bool:
    with h5py.File(dataset, "r") as file:
        first_name = min(file["data"].keys(), key=demo_number)
        return "obs/cameras" in file["data"][first_name]


def validate_target_holes(target: Path, target_names: list[str]) -> None:
    with h5py.File(target, "r") as dataset:
        if "data" not in dataset:
            raise ValueError(f"Target dataset has no /data group: {target}")
        occupied = set(target_names) & set(dataset["data"].keys())
        if occupied:
            raise ValueError(
                "Target episodes must be removed before re-recording; still present: "
                + ", ".join(sorted(occupied, key=demo_number))
            )


def record_replacements(
    *, task: str, replacements: Path, plan: str, count: int, recorder_args: list[str], cameras: bool
) -> None:
    if replacements.exists():
        raise FileExistsError(
            f"Replacement dataset already exists: {replacements}. "
            "Use --use_existing_replacements or choose another path."
        )
    if cameras and "--enable_cameras" not in recorder_args:
        recorder_args = [*recorder_args, "--enable_cameras"]
    if not cameras and "--enable_cameras" in recorder_args:
        raise ValueError("The template has no camera arrays; do not pass --enable_cameras")

    replacements.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RECORDER),
        "--task",
        task,
        "--dataset_file",
        str(replacements),
        "--num_demos",
        str(count),
        "--sobol_plan",
        plan,
        *recorder_args,
    ]
    print(f"Recording exact Sobol plan: {plan}", flush=True)
    print(f"Recorder command: {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def validate_replacements(
    template: Path, replacements: Path, target_names: list[str]
) -> list[tuple[str, str]]:
    with h5py.File(template, "r") as source, h5py.File(replacements, "r") as recorded:
        source_data = source["data"]
        recorded_data = recorded["data"]
        replacement_names = sorted(recorded_data.keys(), key=demo_number)
        if len(replacement_names) != len(target_names):
            raise ValueError(
                f"Expected {len(target_names)} replacements, found {len(replacement_names)}"
            )
        mapping = list(zip(target_names, replacement_names))
        for target_name, replacement_name in mapping:
            expected = source_data[target_name]
            actual = recorded_data[replacement_name]
            if piece_type(actual) != piece_type(expected):
                raise ValueError(
                    f"Piece type mismatch for {target_name}: expected {piece_type(expected)}, "
                    f"got {piece_type(actual)}"
                )
            if not bool(actual.attrs.get("success", False)):
                raise ValueError(f"Replacement {replacement_name} is not marked successful")
            if dataset_paths(actual) != dataset_paths(expected):
                raise ValueError(f"Schema mismatch for {target_name} <- {replacement_name}")
            for path in POSE_PATHS:
                if not np.allclose(
                    episode_translation(actual, path),
                    episode_translation(expected, path),
                    atol=1.0e-5,
                    rtol=0.0,
                ):
                    raise ValueError(f"Move mismatch for {target_name} <- {replacement_name}: {path}")
        return mapping


def refresh_totals(data: h5py.Group) -> None:
    names = sorted(data.keys(), key=demo_number)
    total = sum(int(data[name].attrs["num_samples"]) for name in names)
    data.attrs["num_episodes"] = len(names)
    data.attrs["total"] = total
    data.attrs["total_samples"] = total


def merge_replacements(target: Path, replacements: Path, mapping: list[tuple[str, str]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".tmp.hdf5", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(target, temporary)
        with h5py.File(temporary, "r+") as merged, h5py.File(replacements, "r") as recorded:
            merged_data = merged["data"]
            replacement_data = recorded["data"]
            for target_name, replacement_name in mapping:
                if target_name in merged_data:
                    raise ValueError(f"Refusing to overwrite occupied target group: {target_name}")
                replacement_data.copy(replacement_data[replacement_name], merged_data, name=target_name)
            refresh_totals(merged_data)
            merged.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        with h5py.File(temporary, "r") as merged:
            missing = {name for name, _ in mapping} - set(merged["data"].keys())
            if missing:
                raise RuntimeError(f"Merged file is missing replacements: {sorted(missing)}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args, recorder_args = parse_args()
    template = args.template_dataset.expanduser().resolve()
    target = args.target_dataset.expanduser().resolve()
    replacements = args.replacement_dataset.expanduser().resolve()
    episode_ids = list(dict.fromkeys(args.episode_ids))
    if not template.is_file() or not target.is_file():
        raise FileNotFoundError(f"Template and target must exist: {template}, {target}")

    target_names, plan = build_plan(template, episode_ids)
    validate_target_holes(target, target_names)
    if not args.use_existing_replacements:
        record_replacements(
            task=args.task,
            replacements=replacements,
            plan=plan,
            count=len(target_names),
            recorder_args=recorder_args,
            cameras=records_cameras(template),
        )
    mapping = validate_replacements(template, replacements, target_names)
    merge_replacements(target, replacements, mapping)
    print(f"Filled {len(mapping)} source episodes in {target}")
    for target_name, replacement_name in mapping:
        print(f"  {target_name} <- {replacement_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
