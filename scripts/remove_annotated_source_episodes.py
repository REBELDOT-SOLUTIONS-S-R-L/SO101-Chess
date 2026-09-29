#!/usr/bin/env python3
"""Back up an annotated HDF5 dataset and atomically remove selected demos."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--episode_ids", type=int, nargs="+", required=True)
    return parser.parse_args()


def demo_number(name: str) -> int:
    prefix, separator, suffix = name.rpartition("_")
    if prefix != "demo" or separator != "_" or not suffix.isdigit():
        raise ValueError(f"Unexpected episode group name: {name!r}")
    return int(suffix)


def refresh_totals(data: h5py.Group) -> None:
    names = sorted(data.keys(), key=demo_number)
    total_samples = sum(int(data[name].attrs["num_samples"]) for name in names)
    data.attrs["num_episodes"] = len(names)
    data.attrs["total"] = total_samples
    data.attrs["total_samples"] = total_samples


def validate_dataset(path: Path, required_names: set[str] | None = None) -> None:
    with h5py.File(path, "r") as dataset:
        if "data" not in dataset or not isinstance(dataset["data"], h5py.Group):
            raise ValueError(f"Dataset has no /data group: {path}")
        if required_names:
            missing = sorted(required_names - set(dataset["data"].keys()), key=demo_number)
            if missing:
                raise ValueError(f"Dataset is missing requested episodes: {missing}")


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    backup = args.backup.expanduser().resolve()
    episode_ids = sorted(set(args.episode_ids))
    names = {f"demo_{episode_id}" for episode_id in episode_ids}

    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset}")
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite backup: {backup}")
    validate_dataset(dataset, names)

    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset, backup)
    fsync_file(backup)
    validate_dataset(backup, names)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{dataset.stem}.", suffix=".tmp.hdf5", dir=dataset.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(dataset, temporary)
        with h5py.File(temporary, "r+") as output:
            data = output["data"]
            removed = []
            for name in sorted(names, key=demo_number):
                piece_type = str(data[name].attrs.get("piece_type", ""))
                removed.append((name, piece_type, int(data[name].attrs["num_samples"])))
                del data[name]
            refresh_totals(data)
            output.flush()
        fsync_file(temporary)
        with h5py.File(temporary, "r") as output:
            remaining = set(output["data"].keys())
            if names & remaining:
                raise RuntimeError(f"Temporary dataset still contains removed demos: {sorted(names & remaining)}")
        os.replace(temporary, dataset)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Backup: {backup}")
    print(f"Updated dataset: {dataset}")
    for name, piece_type, samples in removed:
        print(f"Removed {name}: piece_type={piece_type}, samples={samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
