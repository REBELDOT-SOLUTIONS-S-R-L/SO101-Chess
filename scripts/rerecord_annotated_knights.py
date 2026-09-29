#!/usr/bin/env python3
"""Re-record and replace only knight episodes in an annotated chess dataset.

The original dataset is never opened for writing while teleoperation is in
progress. This wrapper records a knight-only replacement dataset with the same
per-type Sobol sequence, verifies that every replacement has the same source
and destination square as its corresponding original episode, and then builds
an updated dataset containing all original non-knight episodes unchanged.

By default the merged dataset is written next to the source with a
``_knight_fixed`` suffix. Pass ``--in_place`` to replace the source atomically;
that mode first creates a ``.before_knight_rerecord.hdf5`` backup.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDER_SCRIPT = REPO_ROOT / "scripts/record_annotated_demos_balanced.py"
DEFAULT_SOURCE_DATASET = REPO_ROOT / "datasets/annotated_dataset_balanced_lowpoly.hdf5"
DEFAULT_TASK = "LeIsaac-SO101-Chess-v0-Mimic"
PIECE_TYPE = "knight"
POSE_PATHS = (
    "initial_state/rigid_objects/active_piece/initial_pose",
    "initial_state/rigid_objects/destination_square/initial_pose",
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Unknown arguments are forwarded to record_annotated_demos_balanced.py.\n"
            "For example: --teleop_device so101leader --port /dev/ttyACM0 --enable_pinocchio"
        ),
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help=f"Isaac Lab task name used for teleoperation. Default: {DEFAULT_TASK}",
    )
    parser.add_argument(
        "--source_dataset",
        type=Path,
        default=DEFAULT_SOURCE_DATASET,
        help="Annotated dataset whose knight episodes will be replaced.",
    )
    parser.add_argument(
        "--output_dataset",
        type=Path,
        default=None,
        help="Merged output path. Defaults to SOURCE_knight_fixed.hdf5.",
    )
    parser.add_argument(
        "--replacement_dataset",
        type=Path,
        default=None,
        help="Temporary knight-only recording path. Defaults beside the source dataset.",
    )
    parser.add_argument(
        "--use_existing_replacements",
        action="store_true",
        help="Skip teleoperation and merge the already-recorded replacement dataset.",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Atomically replace the source dataset after creating a backup.",
    )
    parser.add_argument(
        "--backup_dataset",
        type=Path,
        default=None,
        help="Backup path used with --in_place.",
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Allow replacing an existing non-in-place output dataset.",
    )
    args, recorder_args = parser.parse_known_args()

    forbidden = ("--dataset_file", "--num_demos", "--piece_type")
    conflicting = [
        value
        for value in recorder_args
        if any(value == option or value.startswith(f"{option}=") for option in forbidden)
    ]
    if conflicting:
        parser.error(
            "these recorder options are managed by the knight replacement workflow: "
            + ", ".join(conflicting)
        )
    if "--use_existing_replacements." in recorder_args:
        parser.error("use --use_existing_replacements without a trailing period")
    if args.in_place and args.output_dataset is not None:
        parser.error("--output_dataset cannot be combined with --in_place")
    if args.backup_dataset is not None and not args.in_place:
        parser.error("--backup_dataset requires --in_place")
    return args, recorder_args


def _demo_number(name: str) -> int:
    prefix, separator, suffix = name.rpartition("_")
    if prefix != "demo" or separator != "_" or not suffix.isdigit():
        raise ValueError(f"Unexpected episode group name: {name!r}")
    return int(suffix)


def _sorted_demo_names(data: h5py.Group) -> list[str]:
    return sorted(data.keys(), key=_demo_number)


def _piece_type(episode: h5py.Group) -> str:
    value = episode.attrs.get("piece_type", "")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip().lower()


def _knight_episode_names(dataset_path: Path) -> list[str]:
    with h5py.File(dataset_path, "r") as dataset:
        if "data" not in dataset:
            raise ValueError(f"Dataset has no /data group: {dataset_path}")
        return [
            name
            for name in _sorted_demo_names(dataset["data"])
            if _piece_type(dataset["data"][name]) == PIECE_TYPE
        ]


def _dataset_records_cameras(dataset_path: Path) -> bool:
    with h5py.File(dataset_path, "r") as dataset:
        episode_names = _sorted_demo_names(dataset["data"])
        if not episode_names:
            raise ValueError(f"Dataset contains no episodes: {dataset_path}")
        return "obs/cameras" in dataset["data"][episode_names[0]]


def _dataset_paths(group: h5py.Group) -> set[str]:
    paths: set[str] = set()

    def collect(name: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset):
            paths.add(name)

    group.visititems(collect)
    return paths


def _default_suffixed_path(source: Path, suffix: str) -> Path:
    extension = source.suffix or ".hdf5"
    return source.with_name(f"{source.stem}{suffix}{extension}")


def _inspect_replacement_file(dataset_path: Path) -> tuple[int | None, str | None]:
    """Return the episode count and an error for an existing replacement file."""
    try:
        with h5py.File(dataset_path, "r") as dataset:
            if "data" not in dataset or not isinstance(dataset["data"], h5py.Group):
                return None, "it has no /data group"
            return len(dataset["data"]), None
    except OSError as exc:
        return None, str(exc)


def _archive_incomplete_replacement(dataset_path: Path, reason: str) -> Path:
    """Move an unusable recording aside so a fresh recording can start safely."""
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    archive_path = dataset_path.with_name(
        f"{dataset_path.stem}.incomplete-{timestamp}{dataset_path.suffix}"
    )
    counter = 1
    while archive_path.exists():
        archive_path = dataset_path.with_name(
            f"{dataset_path.stem}.incomplete-{timestamp}-{counter}{dataset_path.suffix}"
        )
        counter += 1
    os.replace(dataset_path, archive_path)
    print(
        f"Archived incomplete replacement dataset to {archive_path}\nReason: {reason}",
        flush=True,
    )
    return archive_path


def _record_replacements(
    *,
    task: str,
    replacement_dataset: Path,
    episode_count: int,
    recorder_args: Iterable[str],
) -> None:
    replacement_dataset.parent.mkdir(parents=True, exist_ok=True)
    if replacement_dataset.exists():
        existing_count, inspection_error = _inspect_replacement_file(replacement_dataset)
        if inspection_error is not None:
            _archive_incomplete_replacement(replacement_dataset, inspection_error)
        elif existing_count is not None and existing_count < episode_count:
            _archive_incomplete_replacement(
                replacement_dataset,
                f"it contains only {existing_count} of {episode_count} required episodes",
            )
        else:
            raise FileExistsError(
                f"Replacement dataset already contains {existing_count} episodes: "
                f"{replacement_dataset}. Pass --use_existing_replacements to validate and merge it, "
                "or choose a different --replacement_dataset."
            )

    command = [
        sys.executable,
        str(RECORDER_SCRIPT),
        "--task",
        task,
        "--dataset_file",
        str(replacement_dataset),
        "--num_demos",
        str(episode_count),
        "--piece_type",
        PIECE_TYPE,
        *recorder_args,
    ]
    print(
        f"Recording {episode_count} knight replacements to {replacement_dataset}\n"
        f"Recorder command: {shlex.join(command)}",
        flush=True,
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _episode_translation(episode: h5py.Group, pose_path: str) -> np.ndarray:
    if pose_path not in episode:
        raise ValueError(f"Episode {episode.name} is missing {pose_path}")
    pose = np.asarray(episode[pose_path][()])
    if pose.shape != (1, 4, 4):
        raise ValueError(
            f"Episode {episode.name}/{pose_path} has shape {pose.shape}; expected (1, 4, 4)"
        )
    return pose[0, :3, 3]


def _validate_replacements(
    source_dataset: Path,
    replacement_dataset: Path,
) -> tuple[list[str], list[str]]:
    with h5py.File(source_dataset, "r") as source, h5py.File(replacement_dataset, "r") as replacements:
        if "data" not in source or "data" not in replacements:
            raise ValueError("Both source and replacement files must contain a /data group")

        source_data = source["data"]
        replacement_data = replacements["data"]
        source_knights = [
            name
            for name in _sorted_demo_names(source_data)
            if _piece_type(source_data[name]) == PIECE_TYPE
        ]
        replacement_names = _sorted_demo_names(replacement_data)

        if not source_knights:
            raise ValueError(f"Source dataset contains no {PIECE_TYPE} episodes")
        if len(replacement_names) != len(source_knights):
            raise ValueError(
                f"Expected {len(source_knights)} replacement episodes, found {len(replacement_names)}"
            )

        for replacement_name in replacement_names:
            episode = replacement_data[replacement_name]
            if _piece_type(episode) != PIECE_TYPE:
                raise ValueError(
                    f"Replacement {replacement_name} has piece_type={_piece_type(episode)!r}; "
                    f"expected {PIECE_TYPE!r}"
                )
            if not bool(episode.attrs.get("success", False)):
                raise ValueError(f"Replacement {replacement_name} is not marked successful")

        for source_name, replacement_name in zip(source_knights, replacement_names):
            source_episode = source_data[source_name]
            replacement_episode = replacement_data[replacement_name]
            source_paths = _dataset_paths(source_episode)
            replacement_paths = _dataset_paths(replacement_episode)
            if replacement_paths != source_paths:
                raise ValueError(
                    f"Schema mismatch for {source_name} <- {replacement_name}: "
                    f"missing={sorted(source_paths - replacement_paths)}, "
                    f"extra={sorted(replacement_paths - source_paths)}"
                )
            for pose_path in POSE_PATHS:
                source_translation = _episode_translation(source_episode, pose_path)
                replacement_translation = _episode_translation(replacement_episode, pose_path)
                if not np.allclose(source_translation, replacement_translation, atol=1.0e-5, rtol=0.0):
                    label = "source" if "active_piece" in pose_path else "destination"
                    raise ValueError(
                        f"Move mismatch for {source_name} <- {replacement_name}: {label} position "
                        f"{replacement_translation.tolist()} does not match "
                        f"{source_translation.tolist()}"
                    )

        return source_knights, replacement_names


def _refresh_dataset_totals(data: h5py.Group) -> None:
    episode_names = _sorted_demo_names(data)
    total_samples = sum(int(data[name].attrs["num_samples"]) for name in episode_names)
    data.attrs["num_episodes"] = len(episode_names)
    data.attrs["total"] = total_samples
    data.attrs["total_samples"] = total_samples


def _build_merged_dataset(
    source_dataset: Path,
    replacement_dataset: Path,
    output_dataset: Path,
    *,
    overwrite_output: bool,
) -> list[tuple[str, str]]:
    source_knights, replacement_names = _validate_replacements(source_dataset, replacement_dataset)
    if output_dataset.exists() and not overwrite_output:
        raise FileExistsError(
            f"Output dataset already exists: {output_dataset}. Pass --overwrite_output to replace it."
        )

    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_dataset.stem}.", suffix=".tmp.hdf5", dir=output_dataset.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source_dataset, temporary_path)
        with (
            h5py.File(temporary_path, "r+") as merged,
            h5py.File(replacement_dataset, "r") as replacements,
        ):
            merged_data = merged["data"]
            replacement_data = replacements["data"]
            for source_name, replacement_name in zip(source_knights, replacement_names):
                del merged_data[source_name]
                replacement_data.copy(replacement_data[replacement_name], merged_data, name=source_name)
            _refresh_dataset_totals(merged_data)
            merged.flush()

        # Re-open the completed temporary file before the atomic rename. This
        # catches truncated or structurally invalid output while the source is
        # still untouched.
        merged_knights = _knight_episode_names(temporary_path)
        if merged_knights != source_knights:
            raise RuntimeError(
                f"Merged knight episode IDs changed: expected {source_knights}, got {merged_knights}"
            )
        with temporary_path.open("rb") as completed_file:
            os.fsync(completed_file.fileno())
        os.replace(temporary_path, output_dataset)
    finally:
        temporary_path.unlink(missing_ok=True)

    return list(zip(source_knights, replacement_names))


def main() -> int:
    args, recorder_args = _parse_args()
    source_dataset = args.source_dataset.expanduser().resolve()
    if not source_dataset.is_file():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dataset}")

    source_knights = _knight_episode_names(source_dataset)
    if not source_knights:
        raise ValueError(f"Source dataset contains no {PIECE_TYPE} episodes: {source_dataset}")

    source_records_cameras = _dataset_records_cameras(source_dataset)
    cameras_requested = "--enable_cameras" in recorder_args
    if cameras_requested and not source_records_cameras:
        raise ValueError(
            "The source dataset does not record cameras; remove --enable_cameras so replacement "
            "episodes retain the same schema."
        )
    if source_records_cameras and not cameras_requested:
        recorder_args.append("--enable_cameras")

    replacement_dataset = (
        args.replacement_dataset.expanduser().resolve()
        if args.replacement_dataset is not None
        else _default_suffixed_path(source_dataset, "_knight_replacements")
    )

    if args.in_place:
        backup_dataset = (
            args.backup_dataset.expanduser().resolve()
            if args.backup_dataset is not None
            else _default_suffixed_path(source_dataset, ".before_knight_rerecord")
        )
        if backup_dataset.exists():
            raise FileExistsError(f"Refusing to overwrite existing backup: {backup_dataset}")
        output_dataset = source_dataset
        overwrite_output = True
    else:
        backup_dataset = None
        output_dataset = (
            args.output_dataset.expanduser().resolve()
            if args.output_dataset is not None
            else _default_suffixed_path(source_dataset, "_knight_fixed")
        )
        overwrite_output = bool(args.overwrite_output)
        if output_dataset.exists() and not overwrite_output:
            raise FileExistsError(
                f"Output dataset already exists: {output_dataset}. "
                "Pass --overwrite_output to replace it."
            )

    if not args.use_existing_replacements:
        _record_replacements(
            task=args.task,
            replacement_dataset=replacement_dataset,
            episode_count=len(source_knights),
            recorder_args=recorder_args,
        )
    elif not replacement_dataset.is_file():
        raise FileNotFoundError(f"Replacement dataset does not exist: {replacement_dataset}")
    else:
        existing_count, inspection_error = _inspect_replacement_file(replacement_dataset)
        if inspection_error is not None:
            raise ValueError(
                f"Replacement dataset is incomplete or unreadable: {replacement_dataset}\n"
                f"Reason: {inspection_error}\n"
                "Rerun without --use_existing_replacements; the file will be archived and a fresh "
                "recording will start."
            )
        if existing_count != len(source_knights):
            raise ValueError(
                f"Replacement dataset contains {existing_count} episodes, but "
                f"{len(source_knights)} are required. Rerun without --use_existing_replacements; "
                "the partial file will be archived and recording will restart."
            )

    # Validate the complete recording before creating an in-place backup or
    # opening any destination file for replacement.
    _validate_replacements(source_dataset, replacement_dataset)
    if args.in_place:
        shutil.copy2(source_dataset, backup_dataset)

    replacements = _build_merged_dataset(
        source_dataset,
        replacement_dataset,
        output_dataset,
        overwrite_output=overwrite_output,
    )
    print(f"Merged dataset written to: {output_dataset}", flush=True)
    if backup_dataset is not None:
        print(f"Original dataset backup: {backup_dataset}", flush=True)
    print(
        "Replaced episodes: "
        + ", ".join(
            f"{source_name}<-{replacement_name}"
            for source_name, replacement_name in replacements
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Knight re-recording failed: {exc}", file=sys.stderr, flush=True)
        raise
