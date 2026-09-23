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

# Importing registers both strategies with IsaacLab Mimic.
from . import pickup_selection as _pickup_selection  # noqa: F401
from .piece_types import normalize_piece_type


class ChessDataGenInfoPool(DataGenInfoPool):
    """Retain explicit piece types and robot joints beside standard DatagenInfo."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.piece_types: list[str] = []
        self.robot_joint_positions: list[torch.Tensor] = []
        self._loading_dataset_file = False

    def load_from_dataset_file(self, file_path, select_demo_keys: str | None = None):
        self._loading_dataset_file = True
        try:
            result = super().load_from_dataset_file(file_path, select_demo_keys=select_demo_keys)
        finally:
            self._loading_dataset_file = False

        source_piece_types = frozenset(self.piece_types)
        if not source_piece_types:
            raise ValueError("The source dataset contains no pickup piece types")
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
        self.piece_types.append(piece_type)
        self.robot_joint_positions.append(joint_positions)


class ChessPickupDataGenerator(DataGenerator):
    """Add type-filtered pickup ownership while retaining standard destination selection."""

    _PICKUP_SUBTASKS = {"pregrasp_align", "grasp", "lift_object"}

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
                current_piece_type=normalize_piece_type(getattr(self.env, "active_piece_name")),
                source_piece_types=self.src_demo_datagen_info_pool.piece_types,
                current_joint_positions=robot.data.joint_pos[env_id, arm_joint_indices],
                source_joint_positions=source_joints,
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
            for sequence in trajectory.waypoint_sequences:
                for waypoint in sequence.sequence:
                    waypoint.pose[:3, 3] += fixed_offset

        from .mdp.observations import (
            get_pickup_source_demo_id,
            set_pickup_source_demo_id,
            set_pregrasp_target_pose,
        )

        selected_demo_id = int(selected_src_demo_inds[eef_name])
        if signal_name == "pregrasp_align":
            set_pickup_source_demo_id(self.env, int(env_id), selected_demo_id)
            open_action = float(self.env_cfg.pickup_open_gripper_action)
            for sequence in trajectory.waypoint_sequences:
                for waypoint in sequence.sequence:
                    waypoint.gripper_action = torch.full_like(waypoint.gripper_action, open_action)
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
            for sequence in trajectory.waypoint_sequences:
                for waypoint in sequence.sequence:
                    waypoint.pose = target_pose.clone()

        return trajectory
