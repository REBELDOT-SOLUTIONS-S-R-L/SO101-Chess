# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Piece-aware source selection strategies for the chess pickup sequence."""

from __future__ import annotations

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab_mimic.datagen.selection_strategy import NearestNeighborMultiObjectStrategy, SelectionStrategy

from .piece_types import normalize_piece_type


def wrapped_angular_distance(current_yaw: torch.Tensor, demo_yaw: torch.Tensor) -> torch.Tensor:
    """Return absolute wrapped yaw error in radians."""
    delta = current_yaw - demo_yaw
    return torch.abs(torch.atan2(torch.sin(delta), torch.cos(delta)))


def _rotation_distance(source_rotations: torch.Tensor, current_rotation: torch.Tensor) -> torch.Tensor:
    """Return geodesic rotation distance from each source rotation to the current rotation."""
    delta = source_rotations @ current_rotation.transpose(-1, -2)
    cosine = (torch.diagonal(delta, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
    return torch.acos(torch.clamp(cosine, -1.0, 1.0))


def _type_filtered_candidate_indices(
    *,
    current_piece_type: str,
    source_piece_types: list[str],
    source_count: int,
    excluded_demo_indices: list[int] | tuple[int, ...] | set[int] | None = None,
) -> list[int]:
    """Return source indices matching one normalized geometry type."""
    if len(source_piece_types) != source_count:
        raise ValueError("source_piece_types must align with the source demonstration pool")

    excluded = {int(index) for index in (excluded_demo_indices or ())}
    invalid = sorted(index for index in excluded if index < 0 or index >= source_count)
    if invalid:
        raise IndexError(
            f"Excluded source demonstration indices are outside the {source_count}-episode pool: {invalid}"
        )

    normalized_type = normalize_piece_type(current_piece_type)
    candidate_indices = [
        index
        for index, piece_type in enumerate(source_piece_types)
        if normalize_piece_type(piece_type) == normalized_type and index not in excluded
    ]
    if not candidate_indices:
        available = sorted(
            {
                normalize_piece_type(piece_type)
                for index, piece_type in enumerate(source_piece_types)
                if index not in excluded
            }
        )
        raise ValueError(
            f"No usable source demonstration has piece_type={normalized_type!r}; "
            f"available types after exclusions: {available}"
        )
    return candidate_indices


class TypeFilteredPickupStrategy(SelectionStrategy):
    """Filter by normalized piece geometry, then rank pickup demonstrations."""

    NAME = "chess_type_filtered_pickup"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        current_piece_type=None,
        source_piece_types=None,
        current_joint_positions=None,
        source_joint_positions=None,
        piece_position_weight=1.0,
        object_rotation_weight=0.05,
        eef_position_weight=1.0,
        eef_rotation_weight=0.05,
        joint_position_weight=0.02,
        knight_yaw_weight=0.1,
        excluded_demo_indices=None,
        nn_k=3,
        **kwargs,
    ):
        """Select from same-type demonstrations using object, EEF and joint state."""
        del kwargs
        if object_pose is None:
            raise ValueError("chess_type_filtered_pickup requires the current active-piece pose")
        if current_piece_type is None or source_piece_types is None:
            raise ValueError("chess_type_filtered_pickup requires current and source piece types")
        if current_joint_positions is None or source_joint_positions is None:
            raise ValueError("chess_type_filtered_pickup requires current and source robot joint positions")
        if len(source_joint_positions) != len(src_subtask_datagen_infos):
            raise ValueError("source_joint_positions must align with the source demonstration pool")

        normalized_type = normalize_piece_type(current_piece_type)
        candidate_indices = _type_filtered_candidate_indices(
            current_piece_type=normalized_type,
            source_piece_types=source_piece_types,
            source_count=len(src_subtask_datagen_infos),
            excluded_demo_indices=excluded_demo_indices,
        )

        device = eef_pose.device
        source_object_poses = torch.stack(
            [next(iter(src_subtask_datagen_infos[index].object_poses.values()))[0] for index in candidate_indices]
        ).to(device)
        source_eef_poses = torch.stack(
            [src_subtask_datagen_infos[index].eef_pose[0] for index in candidate_indices]
        ).to(device)
        source_joints = torch.stack([source_joint_positions[index] for index in candidate_indices]).to(device)
        current_joints = current_joint_positions.to(device).reshape(1, -1)

        source_object_pos, source_object_rot = PoseUtils.unmake_pose(source_object_poses)
        current_object_pos, current_object_rot = PoseUtils.unmake_pose(object_pose)
        source_eef_pos, source_eef_rot = PoseUtils.unmake_pose(source_eef_poses)
        current_eef_pos, current_eef_rot = PoseUtils.unmake_pose(eef_pose)

        score = piece_position_weight * torch.linalg.vector_norm(
            source_object_pos - current_object_pos.reshape(1, 3), dim=-1
        )
        score += object_rotation_weight * _rotation_distance(source_object_rot, current_object_rot)
        score += eef_position_weight * torch.linalg.vector_norm(
            source_eef_pos - current_eef_pos.reshape(1, 3), dim=-1
        )
        score += eef_rotation_weight * _rotation_distance(source_eef_rot, current_eef_rot)
        score += joint_position_weight * torch.linalg.vector_norm(source_joints - current_joints, dim=-1)

        if normalized_type == "knight":
            current_yaw = torch.atan2(current_object_rot[1, 0], current_object_rot[0, 0])
            source_yaw = torch.atan2(source_object_rot[:, 1, 0], source_object_rot[:, 0, 0])
            score += knight_yaw_weight * wrapped_angular_distance(current_yaw, source_yaw)

        top_k = min(max(int(nn_k), 1), len(candidate_indices))
        ordered_local_indices = torch.argsort(score)[:top_k]
        chosen_local_index = int(ordered_local_indices[torch.randint(0, top_k, (1,), device=device)].item())
        return int(candidate_indices[chosen_local_index])


class TypeFilteredMultiObjectStrategy(NearestNeighborMultiObjectStrategy):
    """Run multi-object nearest-neighbor selection within the active piece type."""

    NAME = "chess_type_filtered_multi_object"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        current_piece_type=None,
        source_piece_types=None,
        src_all_object_poses=None,
        **kwargs,
    ):
        if current_piece_type is None or source_piece_types is None:
            raise ValueError(
                "chess_type_filtered_multi_object requires current and source piece types"
            )
        if src_all_object_poses is None:
            raise ValueError(
                "chess_type_filtered_multi_object requires source poses for every compared object"
            )

        candidate_indices = _type_filtered_candidate_indices(
            current_piece_type=current_piece_type,
            source_piece_types=source_piece_types,
            source_count=len(src_subtask_datagen_infos),
        )
        selected_local_index = super().select_source_demo(
            eef_pose=eef_pose,
            object_pose=object_pose,
            src_subtask_datagen_infos=[src_subtask_datagen_infos[index] for index in candidate_indices],
            src_all_object_poses=[src_all_object_poses[index] for index in candidate_indices],
            **kwargs,
        )
        return int(candidate_indices[int(selected_local_index)])


class LatchedPickupStrategy(SelectionStrategy):
    """Return the source episode latched when pregrasp alignment began."""

    NAME = "chess_latched_pickup"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        latched_source_demo_id=None,
        **kwargs,
    ):
        del eef_pose, object_pose, kwargs
        if latched_source_demo_id is None or int(latched_source_demo_id) < 0:
            raise RuntimeError("The pickup source episode is not latched")
        source_demo_id = int(latched_source_demo_id)
        if source_demo_id >= len(src_subtask_datagen_infos):
            raise IndexError(
                f"Latched pickup source {source_demo_id} is outside the "
                f"{len(src_subtask_datagen_infos)}-episode source pool"
            )
        return source_demo_id
