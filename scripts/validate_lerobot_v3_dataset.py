#!/usr/bin/env python3
"""Validate a locally converted LeRobot v3 chess dataset before upload."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq


VIDEO_FEATURES = (
    "observation.images.top_camera",
    "observation.images.right_wrist_camera",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--expected-episodes", required=True, type=int)
    parser.add_argument("--expected-task", required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def parquet_rows(paths: list[Path]) -> int:
    return sum(pq.read_metadata(path).num_rows for path in paths)


def main() -> int:
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(f"Expected LeRobot v3.0, found {info.get('codebase_version')!r}")
    if int(info.get("total_episodes", -1)) != args.expected_episodes:
        raise RuntimeError(
            f"Expected {args.expected_episodes} episodes, found {info.get('total_episodes')}"
        )
    total_frames = int(info.get("total_frames", -1))
    if total_frames <= 0:
        raise RuntimeError(f"Invalid total_frames={total_frames}")
    if float(info.get("fps", 0)) != 60.0:
        raise RuntimeError(f"Expected 60 fps, found {info.get('fps')}")

    features = info.get("features", {})
    for feature in ("observation.state", "action", *VIDEO_FEATURES):
        if feature not in features:
            raise RuntimeError(f"Missing feature {feature!r} from meta/info.json")
    for feature in VIDEO_FEATURES:
        video_info = features[feature]
        if video_info.get("dtype") != "video" or video_info.get("shape") != [480, 640, 3]:
            raise RuntimeError(f"Unexpected video feature metadata for {feature}: {video_info}")

    data_files = sorted((root / "data").rglob("*.parquet"))
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not data_files or not episode_files:
        raise RuntimeError("Missing data or episode Parquet files")
    if parquet_rows(data_files) != total_frames:
        raise RuntimeError("Data Parquet row count does not equal meta/info.json total_frames")
    if parquet_rows(episode_files) != args.expected_episodes:
        raise RuntimeError("Episode Parquet row count does not equal expected episode count")

    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing {tasks_path}")
    tasks = pq.read_table(tasks_path, columns=["task"]).column("task").to_pylist()
    if tasks != [args.expected_task]:
        raise RuntimeError(f"Unexpected tasks metadata: {tasks!r}")

    video_count = 0
    video_bytes = 0
    for feature in VIDEO_FEATURES:
        video_files = sorted((root / "videos" / feature).rglob("*.mp4"))
        if not video_files:
            raise RuntimeError(f"No MP4 files found for {feature}")
        for video_path in video_files:
            if video_path.stat().st_size <= 0:
                raise RuntimeError(f"Empty video file: {video_path}")
            result = subprocess.run(
                [
                    args.ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if float(result.stdout.strip()) <= 0:
                raise RuntimeError(f"Invalid video duration: {video_path}")
            video_count += 1
            video_bytes += video_path.stat().st_size

    print(
        f"Validated {root}: episodes={args.expected_episodes}, frames={total_frames}, "
        f"parquet_files={len(data_files)}, videos={video_count}, "
        f"video_gib={video_bytes / 1024**3:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
