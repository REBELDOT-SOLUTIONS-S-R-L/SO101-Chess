# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder customizations for the single-active-piece chess task.

Isaac Lab's standard initial-state recorder serializes every rigid object in
the scene.  That is useful for general scene replay, but it gives downstream
dataset tools six independent chess-piece streams even though this task has one
logical manipulated object per episode.  The classes here keep the standard
schema while exposing that logical object under the same ``active_piece`` name
used by Mimic datagen observations.
"""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs.mdp.recorders.recorders import StandardInitialStateRecorder
from isaaclab.envs.mdp.recorders.recorders_cfg import StandardInitialStateRecorderCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, StandardHDF5DatasetFileHandler

from .piece_types import normalize_piece_type


ACTIVE_PIECE_DATASET_NAME = "active_piece"
REBELHDF5_SINGLE_ARM_REFERENCE_KEY = "left"


class ActivePieceInitialStateRecorder(StandardInitialStateRecorder):
    """Record only the episode's active piece, using a stable dataset name."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        key, initial_state = super().record_post_reset(env_ids)

        active_piece_name = getattr(self._env, "active_piece_name", None)
        if not isinstance(active_piece_name, str):
            raise RuntimeError(
                "The chess reset event must select env.active_piece_name before "
                "the initial state is recorded."
            )

        rigid_objects = initial_state.get("rigid_objects", {})
        if active_piece_name not in rigid_objects:
            available = sorted(rigid_objects)
            raise KeyError(
                f"Active chess piece {active_piece_name!r} was not found in the recorded scene state. "
                f"Available rigid objects: {available}"
            )

        initial_state["rigid_objects"] = {
            ACTIVE_PIECE_DATASET_NAME: rigid_objects[active_piece_name],
        }
        return key, initial_state


@configclass
class ActivePieceInitialStateRecorderCfg(StandardInitialStateRecorderCfg):
    """Configuration for :class:`ActivePieceInitialStateRecorder`."""

    class_type: type = ActivePieceInitialStateRecorder


class RebelHDF5CompatibleDatasetFileHandler(StandardHDF5DatasetFileHandler):
    """Write standard files plus RebelHDF5's single-arm provenance alias.

    Isaac Lab keys ``reference_demo_indices`` by the configured end-effector
    name.  This task's end effector is named ``chess``.  RebelHDF5 currently
    recognizes only conventional left/right arm keys, so retain the canonical
    ``chess`` dataset and add an identical ``left`` dataset at export time.
    """

    def __init__(self):
        super().__init__()
        self._reference_source_key: str | None = None
        self._env = None

    def set_recorder_metadata(self, cfg, env, failed: bool = False):
        super().set_recorder_metadata(cfg, env, failed=failed)
        self._env = env
        if len(self._entity_order) == 1:
            self._reference_source_key = self._entity_order[0]

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        piece_name = getattr(self._env, "active_piece_name", None)
        if not isinstance(piece_name, str):
            raise RuntimeError("Cannot export a chess demonstration without env.active_piece_name")
        piece_type = normalize_piece_type(piece_name)
        episode_group_name = f"demo_{demo_id}" if demo_id is not None else f"demo_{self._demo_count}"

        references = episode.data.get("reference_demo_indices")
        alias_added = False
        if (
            isinstance(references, dict)
            and self._reference_source_key in references
            and REBELHDF5_SINGLE_ARM_REFERENCE_KEY not in references
        ):
            references[REBELHDF5_SINGLE_ARM_REFERENCE_KEY] = references[self._reference_source_key]
            alias_added = True

        try:
            super().write_episode(episode, demo_id=demo_id)
            # Keep geometry identity explicit even though the manipulated
            # rigid object uses the stable active_piece alias and color is
            # intentionally normalized away.
            self._hdf5_data_group[episode_group_name].attrs["piece_type"] = piece_type
        finally:
            # The alias belongs to the serialized compatibility surface, not
            # the in-memory canonical EpisodeData used by Isaac Lab.
            if alias_added:
                references.pop(REBELHDF5_SINGLE_ARM_REFERENCE_KEY, None)


def configure_chess_recorders(cfg) -> None:
    """Install chess-specific behavior on an annotated or generated recorder.

    The stock Mimic generation launcher replaces the task's recorder config
    after parsing the environment.  Applying this immediately before the
    environment is constructed therefore covers both direct teleoperation and
    stock ``generate_dataset.py --dataset_schema standard`` runs.
    """

    recorder_cfg = getattr(cfg, "recorders", None)
    if recorder_cfg is None:
        return

    initial_state_cfg = getattr(recorder_cfg, "record_initial_state", None)
    if isinstance(initial_state_cfg, StandardInitialStateRecorderCfg):
        recorder_cfg.record_initial_state = ActivePieceInitialStateRecorderCfg()

    handler_type = getattr(recorder_cfg, "dataset_file_handler_class_type", None)
    if isinstance(handler_type, type) and issubclass(handler_type, StandardHDF5DatasetFileHandler):
        recorder_cfg.dataset_file_handler_class_type = RebelHDF5CompatibleDatasetFileHandler
