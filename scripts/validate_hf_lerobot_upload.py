#!/usr/bin/env python3
"""Verify that every local LeRobot dataset file is present in a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-episodes", required=True, type=int)
    return parser.parse_args()


def local_files(root: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[:2] == (".cache", "huggingface"):
            continue
        result[relative.as_posix()] = path.stat().st_size
    return result


def main() -> int:
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    api = HfApi()
    repo = api.repo_info(args.repo_id, repo_type="dataset")
    if repo.private:
        raise RuntimeError(f"Expected a public dataset repository, but {args.repo_id} is private")

    remote: dict[str, int] = {}
    for item in api.list_repo_tree(
        args.repo_id,
        repo_type="dataset",
        recursive=True,
        expand=True,
    ):
        if isinstance(item, RepoFile):
            remote[item.path] = int(item.size)

    local = local_files(root)
    missing = sorted(set(local) - set(remote))
    if missing:
        raise RuntimeError(f"Remote repository is missing local files: {missing[:10]}")
    size_mismatches = sorted(
        path for path, size in local.items() if remote.get(path) != size
    )
    if size_mismatches:
        details = [(path, local[path], remote.get(path)) for path in size_mismatches[:10]]
        raise RuntimeError(f"Remote file size mismatches: {details}")

    with tempfile.TemporaryDirectory(prefix="chess_hf_verify_") as cache_dir:
        info_file = hf_hub_download(
            args.repo_id,
            "meta/info.json",
            repo_type="dataset",
            revision=repo.sha,
            cache_dir=cache_dir,
        )
        info = json.loads(Path(info_file).read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError("Uploaded meta/info.json is not LeRobot v3.0")
    if int(info.get("total_episodes", -1)) != args.expected_episodes:
        raise RuntimeError(
            f"Uploaded metadata has total_episodes={info.get('total_episodes')}, "
            f"expected {args.expected_episodes}"
        )

    print(
        f"Validated Hugging Face upload: repo={args.repo_id}, revision={repo.sha}, "
        f"local_files={len(local)}, remote_files={len(remote)}, "
        f"bytes={sum(local.values())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
