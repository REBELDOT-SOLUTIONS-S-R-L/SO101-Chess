#!/usr/bin/env python3
"""Convert a completed chess HDF5 dataset with RebelHDF5's LeRobot v3 writer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed successful HDF5 dataset")
    parser.add_argument("--output", type=Path, required=True, help="New LeRobot v3 dataset directory")
    parser.add_argument("--modality-json", type=Path, required=True)
    parser.add_argument("--conversion-config", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument(
        "--rebelhdf5-root",
        type=Path,
        default=Path("/home/roboticslab/rebelHDF5"),
        help="RebelHDF5 checkout containing scripts/backend/lerobot.py",
    )
    return parser.parse_args()


def validate_source(path: Path, expected_episodes: int | None) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Source HDF5 does not exist: {path}")

    with h5py.File(path, "r") as dataset:
        if "data" not in dataset:
            raise RuntimeError(f"Source has no /data group: {path}")
        demos = dataset["data"]
        episode_count = len(demos)
        failed = [name for name, demo in demos.items() if not bool(demo.attrs.get("success", False))]
        if failed:
            preview = ", ".join(sorted(failed)[:5])
            raise RuntimeError(
                f"The successful source contains {len(failed)} unsuccessful episodes: {preview}"
            )
        if expected_episodes is not None and episode_count != expected_episodes:
            raise RuntimeError(
                f"Expected {expected_episodes} successful episodes, found {episode_count} in {path}"
            )
    return episode_count


def validate_existing_output(path: Path, expected_episodes: int) -> bool:
    """Return true only when an already-published output is complete and reusable."""
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        return False
    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)
    return info.get("codebase_version") == "v3.0" and info.get("total_episodes") == expected_episodes


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    modality_path = args.modality_json.expanduser().resolve()
    conversion_path = args.conversion_config.expanduser().resolve()
    rebelhdf5_root = args.rebelhdf5_root.expanduser().resolve()

    episode_count = validate_source(input_path, args.expected_episodes)
    print(f"Validated {episode_count} successful HDF5 episodes in {input_path}", flush=True)

    if output_path.exists():
        if validate_existing_output(output_path, episode_count):
            print(f"Complete LeRobot dataset already exists; reusing {output_path}", flush=True)
            return 0
        raise FileExistsError(
            f"Refusing to overwrite incomplete or unexpected output directory: {output_path}"
        )

    scripts_path = rebelhdf5_root / "scripts"
    if not scripts_path.is_dir():
        raise FileNotFoundError(f"RebelHDF5 scripts directory does not exist: {scripts_path}")
    sys.path.insert(0, str(scripts_path))
    from backend.lerobot import convert_with_progress

    last_progress = (-1, "")
    for event in convert_with_progress(
        input_paths=[input_path],
        output_root=output_path,
        modality_json=modality_path,
        conversion_config_json=conversion_path,
        skip_failed=True,
        output_version="v3.0",
        video_codec="h264",
    ):
        event_type = event.get("type")
        if event_type == "progress":
            index = int(event.get("overallDemoIndex", 0))
            count = int(event.get("overallDemoCount", episode_count))
            phase = str(event.get("phase", "conversion"))
            progress = (index, phase)
            if progress != last_progress and (index == count or index % 10 == 0 or phase in {"stats", "metadata"}):
                print(f"LeRobot conversion: {phase} {index}/{count}", flush=True)
                last_progress = progress
        elif event_type == "warning":
            print(f"LeRobot conversion warning: {event.get('message', event)}", flush=True)
        elif event_type == "done":
            print(
                "LeRobot conversion complete: "
                f"{event.get('demoCount')} episodes, {event.get('totalFrames')} frames, "
                f"output={event.get('outputPath')}",
                flush=True,
            )

    if not validate_existing_output(output_path, episode_count):
        raise RuntimeError(f"Converted dataset failed final metadata validation: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
