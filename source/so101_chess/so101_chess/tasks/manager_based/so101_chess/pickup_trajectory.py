# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pose-only adaptations for generated chess pickup trajectories."""

from __future__ import annotations

from collections.abc import Callable

import torch


def clamp_pickup_waypoint_z(trajectory, minimum_z: float | torch.Tensor) -> None:
    """Clamp every pickup waypoint to one minimum robot-root Z coordinate."""
    minimum_z_tensor = torch.as_tensor(minimum_z)
    if minimum_z_tensor.numel() != 1 or not bool(torch.isfinite(minimum_z_tensor).item()):
        raise ValueError("minimum_z must be one finite scalar")

    for sequence in trajectory.waypoint_sequences:
        for waypoint in sequence.sequence:
            pose = waypoint.pose.clone()
            floor = minimum_z_tensor.to(dtype=pose.dtype, device=pose.device).reshape(())
            pose[2, 3] = torch.maximum(pose[2, 3], floor)
            waypoint.pose = pose


def move_trailing_closing_commands_to_grasp(
    boundaries: list[tuple[int, int]],
    gripper_actions: torch.Tensor,
    pregrasp_index: int,
    grasp_index: int,
    open_threshold: float,
) -> list[tuple[int, int]]:
    """Move trailing closing commands from pregrasp into the adjacent grasp segment.

    Recorded termination signals use the measured gripper joint, which lags the
    command. Consequently, the final pregrasp samples may already command the
    gripper to close. Repeating that final sample for measured completion would
    close the gripper before the grasp subtask begins.

    This changes only the boundary between the two segments. Every recorded
    command remains in the generated episode and keeps its original order.
    """
    if not (0 <= pregrasp_index < len(boundaries)):
        raise IndexError(f"Invalid pregrasp boundary index {pregrasp_index}")
    if not (0 <= grasp_index < len(boundaries)):
        raise IndexError(f"Invalid grasp boundary index {grasp_index}")
    if grasp_index != pregrasp_index + 1:
        raise ValueError("pregrasp and grasp boundaries must be adjacent")

    adjusted = list(boundaries)
    pregrasp_start, pregrasp_end = adjusted[pregrasp_index]
    grasp_start, grasp_end = adjusted[grasp_index]
    if grasp_start != pregrasp_end:
        raise ValueError("pregrasp and grasp segments must share a boundary")

    actions = torch.as_tensor(gripper_actions)
    if actions.ndim == 0 or actions.shape[0] < grasp_end:
        raise ValueError("gripper actions do not cover the pickup subtask boundaries")
    command = actions.reshape(actions.shape[0], -1).mean(dim=-1)
    pregrasp_is_open = command[pregrasp_start:pregrasp_end] >= float(open_threshold)
    open_indices = torch.nonzero(pregrasp_is_open, as_tuple=False).flatten()
    if open_indices.numel() == 0:
        raise ValueError(
            "pregrasp segment contains no gripper command at or above the open threshold"
        )

    new_boundary = pregrasp_start + int(open_indices[-1].item()) + 1
    if new_boundary == pregrasp_end:
        return adjusted
    if new_boundary <= pregrasp_start or new_boundary >= grasp_end:
        raise ValueError("adjusted pickup boundary would create an empty subtask")

    adjusted[pregrasp_index] = (pregrasp_start, new_boundary)
    adjusted[grasp_index] = (new_boundary, grasp_end)
    return adjusted


def adjust_pickup_waypoint_poses(
    trajectory,
    fixed_offset: torch.Tensor | None = None,
    held_pose: torch.Tensor | None = None,
) -> None:
    """Adjust pickup poses without changing source gripper commands.

    The source demonstration owns every waypoint's gripper command. Pickup
    customization may translate the TCP or hold the pregrasp pose while the
    source closes the jaw, but must not replace those gripper values.
    """
    if fixed_offset is not None and fixed_offset.shape != (3,):
        raise ValueError(f"fixed_offset must have shape (3,), got {tuple(fixed_offset.shape)}")
    if held_pose is not None and held_pose.shape != (4, 4):
        raise ValueError(f"held_pose must have shape (4, 4), got {tuple(held_pose.shape)}")

    for sequence in trajectory.waypoint_sequences:
        for waypoint in sequence.sequence:
            adjusted_pose = waypoint.pose.clone()
            if fixed_offset is not None:
                adjusted_pose[:3, 3] += fixed_offset
            if held_pose is not None:
                adjusted_pose = held_pose.clone()
            waypoint.pose = adjusted_pose


class MeasuredCompletionWaypointList(list):
    """Repeat the final waypoint until a measured completion predicate latches.

    IsaacLab Mimic advances a subtask when its executable waypoint list is
    exhausted. This list keeps its logical end one step ahead while the
    measured predicate is false and serves the final source waypoint for those
    extra steps. Once the predicate is true, ``len()`` exposes the just-executed
    step as the end of the list so the base generator advances normally.
    """

    def __init__(self, waypoints: list, completion_predicate: Callable[[], bool]):
        if not waypoints:
            raise ValueError("A measured-completion trajectory needs at least one waypoint")
        if not callable(completion_predicate):
            raise TypeError("completion_predicate must be callable")
        super().__init__(waypoints)
        self._source_length = super().__len__()
        self._last_requested_index = -1
        self._completion_predicate = completion_predicate

    def __getitem__(self, index):
        if isinstance(index, slice) or index < 0:
            return super().__getitem__(index)
        self._last_requested_index = max(self._last_requested_index, int(index))
        if index >= self._source_length:
            return super().__getitem__(-1)
        return super().__getitem__(index)

    def __len__(self) -> int:
        # Always execute the complete source segment, even if its measured
        # signal happened to latch early.
        if self._last_requested_index < self._source_length - 1:
            return self._source_length

        if self._completion_predicate():
            return self._last_requested_index + 1

        # Keep the end one step beyond the generator next index. The next
        # lookup repeats the final physical waypoint without growing storage.
        return self._last_requested_index + 2
