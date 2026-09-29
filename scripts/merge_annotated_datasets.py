#!/usr/bin/env python3
"""Append annotated episodes to a base HDF5 dataset without changing base IDs."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import tempfile

import h5py


COUNT_ATTRS = {"num_episodes", "total", "total_samples"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dataset", type=Path, required=True)
    parser.add_argument("--extension_dataset", type=Path, required=True)
    parser.add_argument("--output_dataset", type=Path, required=True)
    parser.add_argument(
        "--expected_extension_per_piece",
        type=int,
        default=None,
        help="Require this many extension episodes for each of the six normalized piece types.",
    )
    return parser.parse_args()


def demo_number(name: str) -> int:
    prefix, separator, suffix = name.rpartition("_")
    if prefix != "demo" or separator != "_" or not suffix.isdigit():
        raise ValueError(f"Unexpected episode group name: {name!r}")
    return int(suffix)


def dataset_paths(group: h5py.Group) -> set[str]:
    paths: set[str] = set()

    def collect(name: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset):
            paths.add(name)

    group.visititems(collect)
    return paths


def normalized_attrs(data: h5py.Group) -> dict[str, object]:
    return {key: data.attrs[key] for key in data.attrs if key not in COUNT_ATTRS}


def piece_type(episode: h5py.Group) -> str:
    value = episode.attrs.get("piece_type", "")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip().lower()


def refresh_totals(data: h5py.Group) -> None:
    names = sorted(data.keys(), key=demo_number)
    total = sum(int(data[name].attrs["num_samples"]) for name in names)
    data.attrs["num_episodes"] = len(names)
    data.attrs["total"] = total
    data.attrs["total_samples"] = total


def validate_compatible(base: h5py.File, extension: h5py.File) -> tuple[list[str], list[str]]:
    if "data" not in base or "data" not in extension:
        raise ValueError("Both files must contain /data groups")
    base_data = base["data"]
    extension_data = extension["data"]
    base_names = sorted(base_data.keys(), key=demo_number)
    extension_names = sorted(extension_data.keys(), key=demo_number)
    if not base_names or not extension_names:
        raise ValueError("Both datasets must contain at least one episode")

    base_attrs = normalized_attrs(base_data)
    extension_attrs = normalized_attrs(extension_data)
    if set(base_attrs) != set(extension_attrs):
        raise ValueError(
            "Dataset metadata keys differ: "
            f"missing={sorted(set(base_attrs) - set(extension_attrs))}, "
            f"extra={sorted(set(extension_attrs) - set(base_attrs))}"
        )
    for key in base_attrs:
        if str(base_attrs[key]) != str(extension_attrs[key]):
            raise ValueError(
                f"Dataset metadata mismatch for {key!r}: "
                f"{base_attrs[key]!r} != {extension_attrs[key]!r}"
            )

    expected_paths = dataset_paths(base_data[base_names[0]])
    for name in extension_names:
        episode = extension_data[name]
        if not bool(episode.attrs.get("success", False)):
            raise ValueError(f"Extension episode is not marked successful: {name}")
        if dataset_paths(episode) != expected_paths:
            raise ValueError(f"Extension episode schema differs from the base: {name}")
    return base_names, extension_names


def main() -> int:
    args = parse_args()
    base = args.base_dataset.expanduser().resolve()
    extension = args.extension_dataset.expanduser().resolve()
    output = args.output_dataset.expanduser().resolve()
    if not base.is_file() or not extension.is_file():
        raise FileNotFoundError(f"Base and extension must exist: {base}, {extension}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if output in {base, extension}:
        raise ValueError("Output must be a new path; the input datasets are never modified")
    if args.expected_extension_per_piece is not None and args.expected_extension_per_piece < 1:
        raise ValueError("--expected_extension_per_piece must be positive")

    with h5py.File(base, "r") as base_file, h5py.File(extension, "r") as extension_file:
        base_names, extension_names = validate_compatible(base_file, extension_file)
        extension_counts = Counter(piece_type(extension_file["data"][name]) for name in extension_names)
        if args.expected_extension_per_piece is not None:
            expected_counts = {
                name: args.expected_extension_per_piece
                for name in ("pawn", "rook", "knight", "bishop", "queen", "king")
            }
            if dict(extension_counts) != expected_counts:
                raise ValueError(
                    "Extension piece counts do not match the requested balanced coverage: "
                    f"expected={expected_counts}, actual={dict(sorted(extension_counts.items()))}"
                )
        next_id = max(demo_number(name) for name in base_names) + 1

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".tmp.hdf5", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    mapping: list[tuple[str, str]] = []
    try:
        shutil.copy2(base, temporary)
        with h5py.File(temporary, "r+") as merged, h5py.File(extension, "r") as extra:
            merged_data = merged["data"]
            extension_data = extra["data"]
            for offset, source_name in enumerate(extension_names):
                target_name = f"demo_{next_id + offset}"
                extension_data.copy(extension_data[source_name], merged_data, name=target_name)
                mapping.append((source_name, target_name))
            refresh_totals(merged_data)
            merged.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        with h5py.File(temporary, "r") as merged:
            expected_count = len(base_names) + len(extension_names)
            if len(merged["data"]) != expected_count:
                raise RuntimeError(
                    f"Merged episode count is {len(merged['data'])}; expected {expected_count}"
                )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Merged {len(extension_names)} episodes into {output}")
    print(f"Extension piece counts: {dict(sorted(extension_counts.items()))}")
    print(f"Episode count: {len(base_names)} -> {len(base_names) + len(extension_names)}")
    print(f"Appended IDs: {mapping[0][1]} .. {mapping[-1][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
