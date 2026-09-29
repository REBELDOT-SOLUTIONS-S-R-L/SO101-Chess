# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Chess-specific Mimic source metadata and pickup trajectory handling."""

from __future__ import annotations

from typing import Any

import torch

from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool

# Importing registers the chess-specific strategies with IsaacLab Mimic.
from . import pickup_selection as _pickup_selection  # noqa: F401
from .piece_types import normalize_piece_type
from .pickup_trajectory import (
    MeasuredCompletionWaypointList,
    adjust_pickup_waypoint_poses,
    clamp_pickup_waypoint_z,
    move_trailing_closing_commands_to_grasp,
)


class ChessDataGenInfoPool(DataGenInfoPool):
    """Retain explicit piece types and robot joints beside standard DatagenInfo."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.piece_types: list[str] = []
        self.robot_joint_positions: list[torch.Tensor] = []
        self.excluded_pickup_demo_indices: frozenset[int] = frozenset()
        self._loading_dataset_file = False

    def load_from_dataset_file(self, file_path, select_demo_keys: str | None = None):
        self._loading_dataset_file = True
        try:
            result = super().load_from_dataset_file(file_path, select_demo_keys=select_demo_keys)
        finally:
            self._loading_dataset_file = False

        excluded_specs = tuple(getattr(self.env_cfg, "pickup_excluded_sources", ()))
        excluded_indices: set[int] = set()
        for demo_index, expected_piece_type in excluded_specs:
            demo_index = int(demo_index)
            if demo_index < 0 or demo_index >= len(self.piece_types):
                raise IndexError(
                    f"Excluded pickup source {demo_index} is outside the "
                    f"{len(self.piece_types)}-episode source dataset"
                )
            actual_piece_type = normalize_piece_type(self.piece_types[demo_index])
            expected_piece_type = normalize_piece_type(expected_piece_type)
            if actual_piece_type != expected_piece_type:
                raise ValueError(
                    f"Excluded pickup source {demo_index} was expected to be "
                    f"{expected_piece_type!r}, but the loaded dataset contains {actual_piece_type!r}. "
                    "Update pickup_excluded_sources for this source dataset."
                )
            excluded_indices.add(demo_index)
        self.excluded_pickup_demo_indices = frozenset(excluded_indices)

        source_piece_types = frozenset(
            piece_type
            for demo_index, piece_type in enumerate(self.piece_types)
            if demo_index not in self.excluded_pickup_demo_indices
        )
        if not source_piece_types:
            raise ValueError("The source dataset contains no usable pickup piece types after exclusions")
        # The next generation reset must sample only types for which strict
        # type-filtered pickup has at least one source demonstration.
        self.env._chess_pickup_source_piece_types = source_piece_types
        return result

    def _add_episode(self, episode):
        root_attrs = episode.attrs.get("", {}) if isinstance(episode.attrs, dict) else {}
        piece_type = root_attrs.get("piece_type")
        if isinstance(piece_type, bytes):
            piece_type = piece_type.decode("utf-8")
        if piece_type is None and not self._loading_dataset_file:
            piece_type = getattr(self.env, "active_piece_name", None)
        if piece_type is None:
            raise ValueError(
                "Source demonstration is missing its required per-episode 'piece_type' attribute. "
                "Record or migrate the dataset with the chess recorder before type-filtered pickup generation."
            )
        piece_type = normalize_piece_type(piece_type)

        observations = episode.data.get("obs", {})
        joint_positions = None
        articulations = observations.get("articulations", {}) if isinstance(observations, dict) else {}
        robot_observations = articulations.get("robot", {}) if isinstance(articulations, dict) else {}
        if isinstance(robot_observations, dict):
            joint_positions = robot_observations.get("joint_position")
        if joint_positions is None and isinstance(observations, dict):
            joint_positions = observations.get("robot_joint_pos")
        if joint_positions is None:
            raise ValueError(
                "Source demonstration is missing robot joint positions required by pickup nearest-neighbor selection."
            )

        previous_count = len(self.datagen_infos)
        super()._add_episode(episode)
        if len(self.datagen_infos) != previous_count + 1:
            raise RuntimeError("DataGenInfo pool did not append exactly one source demonstration")

        for eef_name, signal_names in self.subtask_term_signal_names.items():
            if "pregrasp_align" not in signal_names or "grasp" not in signal_names:
                continue
            pregrasp_index = signal_names.index("pregrasp_align")
            grasp_index = signal_names.index("grasp")
            pregrasp_term = self.env_cfg.observations.subtask_terms.pregrasp_align
            open_threshold = float(pregrasp_term.params["gripper_open_threshold"])
            episode_boundaries = self._subtask_boundaries[eef_name][-1]
            gripper_actions = self.datagen_infos[-1].gripper_action[eef_name]
            self._subtask_boundaries[eef_name][-1] = move_trailing_closing_commands_to_grasp(
                boundaries=episode_boundaries,
                gripper_actions=gripper_actions,
                pregrasp_index=pregrasp_index,
                grasp_index=grasp_index,
                open_threshold=open_threshold,
            )

        self.piece_types.append(piece_type)
        self.robot_joint_positions.append(joint_positions)


class ChessPickupDataGenerator(DataGenerator):
    """Add type-filtered source selection and latched pickup ownership."""

    _PICKUP_SUBTASKS = {"pregrasp_align", "grasp", "lift_object"}
    _MEASURED_COMPLETION_SUBTASKS = {"pregrasp_align", "grasp"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.src_demo_datagen_info_pool, ChessDataGenInfoPool):
            raise TypeError("ChessPickupDataGenerator requires ChessDataGenInfoPool")
        self._selection_context: dict[str, Any] | None = None

    def select_source_demo(
        self,
        eef_name,
        eef_pose,
        object_pose,
        src_demo_current_subtask_boundaries,
        subtask_object_name,
        selection_strategy_name,
        selection_strategy_kwargs=None,
        all_object_poses=None,
        source_demo_selections=None,
    ):
        kwargs = dict(selection_strategy_kwargs or {})
        context = self._selection_context
        if context is None:
            raise RuntimeError("Chess pickup selection was called without an active environment context")

        env_id = int(context["env_id"])
        if selection_strategy_name in {
            "chess_type_filtered_pickup",
            "chess_type_filtered_multi_object",
        }:
            kwargs.update(
                current_piece_type=normalize_piece_type(getattr(self.env, "active_piece_name")),
                source_piece_types=self.src_demo_datagen_info_pool.piece_types,
            )

        if selection_strategy_name == "chess_type_filtered_pickup":
            robot = self.env.scene["robot"]
            gripper_joint_indices, _ = robot.find_joints("gripper")
            arm_joint_indices = [
                index for index in range(robot.data.joint_pos.shape[-1]) if index not in gripper_joint_indices
            ]
            if not arm_joint_indices:
                raise RuntimeError("No arm joints remain after excluding the gripper from pickup selection")

            source_joints = []
            for demo_index, boundary in enumerate(src_demo_current_subtask_boundaries):
                start_index = int(boundary[0])
                source_joints.append(
                    self.src_demo_datagen_info_pool.robot_joint_positions[demo_index][
                        start_index, arm_joint_indices
                    ]
                )
            kwargs.update(
                current_joint_positions=robot.data.joint_pos[env_id, arm_joint_indices],
                source_joint_positions=source_joints,
                excluded_demo_indices=self.src_demo_datagen_info_pool.excluded_pickup_demo_indices,
            )
        elif selection_strategy_name == "chess_latched_pickup":
            from .mdp.observations import get_pickup_source_demo_id

            kwargs["latched_source_demo_id"] = get_pickup_source_demo_id(self.env, env_id)

        return super().select_source_demo(
            eef_name=eef_name,
            eef_pose=eef_pose,
            object_pose=object_pose,
            src_demo_current_subtask_boundaries=src_demo_current_subtask_boundaries,
            subtask_object_name=subtask_object_name,
            selection_strategy_name=selection_strategy_name,
            selection_strategy_kwargs=kwargs,
            all_object_poses=all_object_poses,
            source_demo_selections=source_demo_selections,
        )

    def _subtask_signal_latched(self, env_id: int, signal_name: str) -> bool:
        signals = self.env.get_subtask_term_signals(env_ids=[env_id])
        if signal_name not in signals:
            raise KeyError(f"Missing measured subtask signal {signal_name!r}")
        signal = torch.as_tensor(signals[signal_name]).reshape(-1)
        if signal.numel() != 1:
            raise RuntimeError(
                f"Expected one {signal_name!r} value for env {env_id}, got shape "
                f"{tuple(signal.shape)}"
            )
        return bool(signal[0].item())

    def merge_eef_subtask_trajectory(
        self,
        env_id: int,
        eef_name: str,
        subtask_index: int,
        prev_executed_traj,
        subtask_trajectory,
    ):
        waypoints = super().merge_eef_subtask_trajectory(
            env_id=env_id,
            eef_name=eef_name,
            subtask_index=subtask_index,
            prev_executed_traj=prev_executed_traj,
            subtask_trajectory=subtask_trajectory,
        )
        signal_name = self.env_cfg.subtask_configs[eef_name][subtask_index].subtask_term_signal
        if signal_name not in self._MEASURED_COMPLETION_SUBTASKS:
            return waypoints

        return MeasuredCompletionWaypointList(
            waypoints,
            completion_predicate=lambda: self._subtask_signal_latched(int(env_id), signal_name),
        )

    def generate_eef_subtask_trajectory(
        self,
        env_id,
        eef_name,
        subtask_ind,
        all_randomized_subtask_boundaries,
        runtime_subtask_constraints_dict,
        selected_src_demo_inds,
        source_demo_selections=None,
    ):
        signal_name = self.env_cfg.subtask_configs[eef_name][subtask_ind].subtask_term_signal
        self._selection_context = {"env_id": int(env_id), "subtask_ind": int(subtask_ind)}
        try:
            trajectory = super().generate_eef_subtask_trajectory(
                env_id=env_id,
                eef_name=eef_name,
                subtask_ind=subtask_ind,
                all_randomized_subtask_boundaries=all_randomized_subtask_boundaries,
                runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
                selected_src_demo_inds=selected_src_demo_inds,
                source_demo_selections=source_demo_selections,
            )
        finally:
            self._selection_context = None

        if signal_name not in self._PICKUP_SUBTASKS:
            return trajectory

        fixed_offset = torch.tensor(
            self.env_cfg.pickup_tcp_offset,
            dtype=trajectory[0].pose.dtype,
            device=trajectory[0].pose.device,
        )
        if bool(torch.any(fixed_offset != 0).item()):
            adjust_pickup_waypoint_poses(trajectory, fixed_offset=fixed_offset)

        active_piece_pose = self.env.get_object_poses(env_ids=[int(env_id)]).get("active_piece")
        if active_piece_pose is None or active_piece_pose.shape != (1, 4, 4):
            actual_shape = None if active_piece_pose is None else tuple(active_piece_pose.shape)
            raise RuntimeError(
                "Pickup Z clamping requires one active-piece pose with shape (1, 4, 4); "
                f"got {actual_shape}"
            )
        minimum_z = active_piece_pose[0, 2, 3] + float(self.env_cfg.pickup_min_tcp_clearance)
        clamp_pickup_waypoint_z(trajectory, minimum_z)

        from .mdp.observations import (
            get_pickup_source_demo_id,
            set_pickup_source_demo_id,
            set_pregrasp_target_pose,
        )

        selected_demo_id = int(selected_src_demo_inds[eef_name])
        if signal_name == "pregrasp_align":
            set_pickup_source_demo_id(self.env, int(env_id), selected_demo_id)
            set_pregrasp_target_pose(self.env, int(env_id), trajectory.last_waypoint.pose)
            return trajectory

        latched_demo_id = get_pickup_source_demo_id(self.env, int(env_id))
        if selected_demo_id != latched_demo_id:
            raise RuntimeError(
                f"Pickup source changed during {signal_name}: selected {selected_demo_id}, "
                f"latched {latched_demo_id}."
            )

        if signal_name == "grasp":
            target_pose = self.env._chess_pregrasp_target_pose[int(env_id)].clone()
            adjust_pickup_waypoint_poses(trajectory, held_pose=target_pose)

        return trajectory
