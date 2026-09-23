# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific observation functions for so101_chess.

These helpers implement the subtask termination predicates used by the Mimic
annotated recorder for the chess move sequence:
    move_over_source -> pregrasp_align -> grasp -> lift_object ->
    move_over_destination -> place_object -> lift_after_place -> return_home
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul

from .board import square_surface_position

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_active_piece_name(env: ManagerBasedRLEnv) -> str:
    """Return the current episode's active piece or a safe fallback."""
    piece_name = getattr(env, "active_piece_name", None)
    if isinstance(piece_name, str) and piece_name in env.scene.keys():
        return piece_name
    for candidate in ("bishop_white", "pawn_white", "rook_white", "knight_white", "queen_white", "king_white"):
        if candidate in env.scene.keys():
            return candidate
    raise RuntimeError("No chess piece asset is available in the scene.")


def _square_center_world(env: ManagerBasedRLEnv, square: tuple[int, int] | list[int]) -> torch.Tensor:
    """Return the 3D center of a board square in world coordinates."""
    row, col = int(square[0]), int(square[1])
    if not (0 <= row <= 7 and 0 <= col <= 7):
        raise ValueError(f"Invalid chess square: {square}")

    # The board is static for the lifetime of an environment. Subtask
    # observations are evaluated more than once per control step by the
    # recorder, so avoid rebuilding USD transform/bounds caches each time.
    cache = getattr(env, "_chess_square_position_cache", None)
    if cache is None:
        cache = {}
        env._chess_square_position_cache = cache
    cache_key = (row, col)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    stage = env.sim.stage
    env_id_found = 0
    for env_id in range(env.num_envs):
        prim_path = f"/World/envs/env_{env_id}/Scene/ChessBoard/Square_{row}_{col}"
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            env_id_found = env_id
            break
    else:
        raise RuntimeError(f"Square prim not found for square {(row, col)}")

    position = square_surface_position(stage, env_id_found, row, col, z_offset=0.001)
    center = torch.tensor(
        tuple(position),
        dtype=torch.float,
        device=env.device,
    )
    cache[cache_key] = center
    return center


def _source_target_world(env: ManagerBasedRLEnv, square_name: str) -> torch.Tensor:
    """Return the world position of the current source or target board square."""
    source = getattr(env, "active_source_square", None)
    target = getattr(env, "active_target_square", None)
    if square_name == "source" and source is not None:
        return _square_center_world(env, source)
    if square_name == "target" and target is not None:
        return _square_center_world(env, target)

    if square_name == "source":
        return _square_center_world(env, (2, 2))
    return _square_center_world(env, (3, 3))


DEBUG_SUBTASK_LOG = False
_LAST_SUBTASK_LOG_TS = 0.0

_SUBTASK_ORDER = (
    "move_over_source",
    "pregrasp_align",
    "grasp",
    "lift_object",
    "move_over_destination",
    "place_object",
    "lift_after_place",
    "return_home",
)


def _ensure_pickup_state(env: ManagerBasedRLEnv) -> None:
    """Create per-environment pickup ownership and pose-stability buffers."""
    if not hasattr(env, "_chess_pickup_source_demo_ids"):
        env._chess_pickup_source_demo_ids = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "_chess_pregrasp_target_pose"):
        env._chess_pregrasp_target_pose = torch.eye(
            4, dtype=torch.float32, device=env.device
        ).unsqueeze(0).repeat(env.num_envs, 1, 1)
    if not hasattr(env, "_chess_pregrasp_target_valid"):
        env._chess_pregrasp_target_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "_chess_pregrasp_target_external"):
        env._chess_pregrasp_target_external = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "_chess_pregrasp_hold_counter"):
        env._chess_pregrasp_hold_counter = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "_chess_pregrasp_prev_pose"):
        env._chess_pregrasp_prev_pose = torch.eye(
            4, dtype=torch.float32, device=env.device
        ).unsqueeze(0).repeat(env.num_envs, 1, 1)
    if not hasattr(env, "_chess_pregrasp_last_counted_step"):
        env._chess_pregrasp_last_counted_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "_chess_grasp_hold_counter"):
        env._chess_grasp_hold_counter = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "_chess_grasp_prev_grip"):
        env._chess_grasp_prev_grip = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "_chess_grasp_prev_pose"):
        env._chess_grasp_prev_pose = torch.eye(
            4, dtype=torch.float32, device=env.device
        ).unsqueeze(0).repeat(env.num_envs, 1, 1)
    if not hasattr(env, "_chess_grasp_last_counted_step"):
        env._chess_grasp_last_counted_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )


def set_pickup_source_demo_id(env: ManagerBasedRLEnv, env_id: int, source_demo_id: int) -> None:
    """Latch the source episode that owns pregrasp, grasp and lift."""
    _ensure_pickup_state(env)
    current = int(env._chess_pickup_source_demo_ids[env_id].item())
    if current >= 0 and current != int(source_demo_id):
        raise RuntimeError(f"Pickup source is already latched to {current}, cannot replace it with {source_demo_id}")
    env._chess_pickup_source_demo_ids[env_id] = int(source_demo_id)


def get_pickup_source_demo_id(env: ManagerBasedRLEnv, env_id: int) -> int:
    """Return the active pickup source episode, or -1 when none is owned."""
    _ensure_pickup_state(env)
    return int(env._chess_pickup_source_demo_ids[env_id].item())


def set_pregrasp_target_pose(env: ManagerBasedRLEnv, env_id: int, target_pose: torch.Tensor) -> None:
    """Store the transformed demonstrated grasp pose for measured-pose checks."""
    _ensure_pickup_state(env)
    env._chess_pregrasp_target_pose[env_id] = target_pose.to(
        device=env.device, dtype=env._chess_pregrasp_target_pose.dtype
    )
    env._chess_pregrasp_target_valid[env_id] = True
    env._chess_pregrasp_target_external[env_id] = True
    env._chess_pregrasp_hold_counter[env_id] = 0
    env._chess_grasp_hold_counter[env_id] = 0


def clear_pickup_state(env: ManagerBasedRLEnv, env_ids) -> None:
    """Clear pickup ownership after lift completion, reset, or an aborted attempt."""
    _ensure_pickup_state(env)
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).reshape(-1)
    if env_ids.numel() == 0:
        return
    env._chess_pickup_source_demo_ids[env_ids] = -1
    env._chess_pregrasp_target_valid[env_ids] = False
    env._chess_pregrasp_target_external[env_ids] = False
    env._chess_pregrasp_hold_counter[env_ids] = 0
    env._chess_pregrasp_last_counted_step[env_ids] = -1
    env._chess_grasp_hold_counter[env_ids] = 0
    env._chess_grasp_last_counted_step[env_ids] = -1


def _pose_error(current: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return position and geodesic orientation error for batched poses."""
    position_error = torch.linalg.vector_norm(current[:, :3, 3] - target[:, :3, 3], dim=-1)
    delta = current[:, :3, :3] @ target[:, :3, :3].transpose(-1, -2)
    cosine = (torch.diagonal(delta, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
    orientation_error = torch.acos(torch.clamp(cosine, -1.0, 1.0))
    return position_error, orientation_error


def _mark_chess_failed(env: ManagerBasedRLEnv, mask: torch.Tensor) -> None:
    """OR ``mask`` into the per-env chess-move fail latch read by ``chess_move_failed``."""
    if not hasattr(env, "_chess_subtask_failed"):
        env._chess_subtask_failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._chess_subtask_failed |= mask
    failed_env_ids = torch.nonzero(mask, as_tuple=False).flatten()
    if failed_env_ids.numel() > 0:
        clear_pickup_state(env, failed_env_ids)


def get_chess_subtask_failed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the latched per-env chess-move fail flag set by the subtask done() functions."""
    return getattr(env, "_chess_subtask_failed", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))


def _is_stage_pending(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """True where ``name`` is the currently active (not-yet-latched) subtask."""
    stage_idx = _SUBTASK_ORDER.index(name)
    if not hasattr(env, "_chess_subtask_stage"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env._chess_subtask_stage == stage_idx


def _subtask_gate(
    env: ManagerBasedRLEnv,
    name: str,
    raw_done: torch.Tensor,
    max_steps: int | None = None,
) -> torch.Tensor:
    """Gate + latch a subtask signal so it can only fire once all prior subtasks
    in ``_SUBTASK_ORDER`` have already latched; blocks any out-of-order/false-positive fire.

    If ``max_steps`` is given, stalling on this subtask for more than that many
    consecutive steps marks the episode as failed via ``_mark_chess_failed``.
    """
    stage_idx = _SUBTASK_ORDER.index(name)

    if not hasattr(env, "_chess_subtask_stage"):
        env._chess_subtask_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_chess_subtask_stage_steps"):
        env._chess_subtask_stage_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_chess_subtask_last_counted_step"):
        env._chess_subtask_last_counted_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )

    stage = env._chess_subtask_stage
    stage_steps = env._chess_subtask_stage_steps
    last_counted_step = env._chess_subtask_last_counted_step
    raw_bool = raw_done.squeeze(-1).bool() if raw_done.dim() > 1 else raw_done.bool()

    prereq_met = stage >= stage_idx
    gated = raw_bool & prereq_met

    advance = gated & (stage == stage_idx)
    stage[advance] = stage_idx + 1

    # Observation groups may be evaluated more than once during one env step.
    # Count a pending subtask only once for each episode step.
    is_pending = stage == stage_idx
    episode_step = env.episode_length_buf.to(device=env.device, dtype=torch.long)
    count_step = is_pending & (last_counted_step != episode_step)
    stage_steps[count_step] += 1
    last_counted_step[count_step] = episode_step[count_step]
    stage_steps[advance] = 0
    last_counted_step[advance] = -1

    if max_steps is not None:
        _mark_chess_failed(env, is_pending & (stage_steps > max_steps))

    latched = stage > stage_idx
    return latched.float().unsqueeze(-1)


def reset_chess_subtask_stage(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset the sequential subtask gate for the given env ids on episode reset."""
    if hasattr(env, "_chess_subtask_stage"):
        env._chess_subtask_stage[env_ids] = 0
    if hasattr(env, "_chess_subtask_stage_steps"):
        env._chess_subtask_stage_steps[env_ids] = 0
    if hasattr(env, "_chess_subtask_last_counted_step"):
        env._chess_subtask_last_counted_step[env_ids] = -1
    if hasattr(env, "_chess_subtask_failed"):
        env._chess_subtask_failed[env_ids] = False
    clear_pickup_state(env, env_ids)
    if hasattr(env, "_chess_post_place_lift_initialized"):
        env._chess_post_place_lift_initialized[env_ids] = False
    if hasattr(env, "_chess_post_place_lift_start"):
        env._chess_post_place_lift_start[env_ids] = 0
    if hasattr(env, "_chess_home_hold_counter"):
        env._chess_home_hold_counter[env_ids] = 0
    if hasattr(env, "_chess_home_last_counted_step"):
        env._chess_home_last_counted_step[env_ids] = -1
    if hasattr(env, "_chess_home_prev_eef"):
        env._chess_home_prev_eef[env_ids] = 0


def _gripper_joint_mean(env: ManagerBasedRLEnv, joint_pattern: str = "gripper") -> torch.Tensor:
    """Mean gripper joint value over joints matching the pattern."""
    robot = env.scene["robot"]
    cache = getattr(env, "_chess_gripper_joint_index_cache", None)
    if cache is None:
        cache = {}
        env._chess_gripper_joint_index_cache = cache
    idx_tensor = cache.get(joint_pattern)
    if idx_tensor is None:
        indexes, _ = robot.find_joints(joint_pattern)
        if not indexes:
            return torch.zeros(env.num_envs, device=env.device)
        idx_tensor = torch.tensor(indexes, dtype=torch.long, device=env.device)
        cache[joint_pattern] = idx_tensor
    return robot.data.joint_pos[:, idx_tensor].mean(dim=1)


def _eef_position(env: ManagerBasedRLEnv, eef_link: str) -> torch.Tensor:
    """Return the end-effector position relative to each environment origin."""
    robot = env.scene["robot"]
    body_index = robot.data.body_names.index(eef_link)
    return robot.data.body_pos_w[:, body_index] - env.scene.env_origins


def _eef_position_robot_root(env: ManagerBasedRLEnv, eef_link: str) -> torch.Tensor:
    """Return the end-effector position expressed in the robot-root frame."""
    robot = env.scene["robot"]
    body_index = robot.data.body_names.index(eef_link)
    root_to_eef_w = robot.data.body_pos_w[:, body_index] - robot.data.root_pos_w
    return quat_apply_inverse(robot.data.root_quat_w, root_to_eef_w)


def _eef_pose_robot_root(env: ManagerBasedRLEnv, eef_link: str) -> torch.Tensor:
    """Return the measured end-effector pose in the robot-root frame."""
    robot = env.scene["robot"]
    body_index = robot.data.body_names.index(eef_link)
    root_to_eef_w = robot.data.body_pos_w[:, body_index] - robot.data.root_pos_w
    position = quat_apply_inverse(robot.data.root_quat_w, root_to_eef_w)
    orientation = quat_mul(quat_inv(robot.data.root_quat_w), robot.data.body_quat_w[:, body_index])
    pose = torch.eye(4, dtype=position.dtype, device=position.device).unsqueeze(0).repeat(env.num_envs, 1, 1)
    pose[:, :3, :3] = matrix_from_quat(orientation)
    pose[:, :3, 3] = position
    return pose


def _piece_upright(piece_quat_w: torch.Tensor, max_tilt_deg: float) -> torch.Tensor:
    """Return whether each piece's local up axis is within ``max_tilt_deg`` of world up."""
    local_up = torch.tensor(
        [0.0, 0.0, 1.0], device=piece_quat_w.device, dtype=piece_quat_w.dtype
    ).expand(piece_quat_w.shape[0], 3)
    piece_up = quat_apply(piece_quat_w, local_up)
    return piece_up[:, 2] >= math.cos(math.radians(max_tilt_deg))


def _piece_on_square(
    piece_position: torch.Tensor,
    piece_quat_w: torch.Tensor,
    square_position: torch.Tensor,
    xy_threshold: float,
    z_threshold: float,
    max_tilt_deg: float,
) -> torch.Tensor:
    """Return whether each piece is centered, supported, and upright on a square."""
    xy_distance = torch.linalg.vector_norm(piece_position[:, :2] - square_position[:, :2], dim=-1)
    z_distance = torch.abs(piece_position[:, 2] - square_position[:, 2])
    return (
        (xy_distance <= xy_threshold)
        & (z_distance <= z_threshold)
        & _piece_upright(piece_quat_w, max_tilt_deg)
    )


def _debug_print_subtask(
    env: ManagerBasedRLEnv,
    signal_name: str | None,
    name: str,
    actual_value: float,
    trigger_value: float,
    unit: str = "",
) -> None:
    """Log only the active subtask, once every ~3s, so the terminal stays readable."""
    global _LAST_SUBTASK_LOG_TS
    if not DEBUG_SUBTASK_LOG or env.num_envs <= 0:
        return
    if signal_name is not None and signal_name != name:
        return
    now = time.monotonic()
    if now - _LAST_SUBTASK_LOG_TS < 3.0:
        return
    _LAST_SUBTASK_LOG_TS = now
    print(f"[chess-subtask] {name}: actual={actual_value:.4f}{unit}, trigger={trigger_value:.4f}{unit}")


def move_over_source_done(
    env: ManagerBasedRLEnv,
    eef_link: str,
    object_name: str | None = None,
    hover_z_offset: float = 0.10,
    hover_xy_tolerance: float = 0.025,
    hover_z_tolerance: float = 0.015,
    gripper_joint_pattern: str = "gripper",
    gripper_open_threshold: float = 0.5,
    piece_source_xy_threshold: float = 0.05,
    piece_source_z_threshold: float = 0.03,
    piece_max_tilt_deg: float = 30.0,
    max_steps: int = 300,
    signal_name: str | None = None,
) -> torch.Tensor:
    """Latch only at the defined open-gripper hover pose above the active piece."""
    if hover_z_offset <= 0.0:
        raise ValueError("hover_z_offset must be positive")

    eef = _eef_position(env, eef_link)
    source = _source_target_world(env, "source").unsqueeze(0).expand_as(eef)
    if object_name is None:
        object_name = _get_active_piece_name(env)
    piece = env.scene[object_name]
    obj = piece.data.root_pos_w - env.scene.env_origins

    # Center the hover target on the measured piece XY while keeping a fixed,
    # geometry-independent clearance above the board surface.
    hover = source.clone()
    hover[:, :2] = obj[:, :2]
    hover[:, 2] += hover_z_offset
    xy_error = torch.linalg.vector_norm(eef[:, :2] - hover[:, :2], dim=-1)
    z_error = torch.abs(eef[:, 2] - hover[:, 2])
    grip = _gripper_joint_mean(env, gripper_joint_pattern)
    open_grip = grip >= gripper_open_threshold
    done = (
        (xy_error <= hover_xy_tolerance)
        & (z_error <= hover_z_tolerance)
        & open_grip
    ).float().unsqueeze(-1)
    _debug_print_subtask(
        env, signal_name, "move_over_source", float(z_error[0].item()), float(hover_z_tolerance), unit="m"
    )

    piece_ready = _piece_on_square(
        obj, piece.data.root_quat_w, source, piece_source_xy_threshold, piece_source_z_threshold, piece_max_tilt_deg
    )
    _mark_chess_failed(env, _is_stage_pending(env, "move_over_source") & ~piece_ready)
    return _subtask_gate(env, "move_over_source", done, max_steps=max_steps)


def pregrasp_align_done(
    env: ManagerBasedRLEnv,
    eef_link: str,
    object_name: str | None = None,
    gripper_joint_pattern: str = "gripper",
    gripper_open_threshold: float = 0.3,
    piece_dist_threshold: float = 0.08,
    position_tolerance: float = 0.005,
    orientation_tolerance: float = 0.05,
    max_position_step: float = 0.0015,
    max_orientation_step: float = 0.03,
    stable_frames: int = 5,
    recording_piece_dist_threshold: float = 0.05,
    recording_position_tolerance: float = 0.002,
    recording_orientation_tolerance: float = 0.02,
    recording_max_position_step: float = 0.00075,
    recording_max_orientation_step: float = 0.01,
    recording_stable_frames: int = 10,
    piece_source_xy_threshold: float = 0.05,
    piece_source_z_threshold: float = 0.03,
    piece_max_tilt_deg: float = 30.0,
    max_steps: int = 300,
    signal_name: str | None = None,
) -> torch.Tensor:
    """Capture a stable recording pose or reach the selected generation target with the gripper open."""
    if stable_frames < 1 or recording_stable_frames < 1:
        raise ValueError("stable frame requirements must be at least one")

    pose = _eef_pose_robot_root(env, eef_link)
    eef = _eef_position(env, eef_link)
    _ensure_pickup_state(env)
    if env._chess_pregrasp_target_pose.dtype != pose.dtype:
        env._chess_pregrasp_target_pose = env._chess_pregrasp_target_pose.to(dtype=pose.dtype)

    if object_name is None:
        object_name = _get_active_piece_name(env)
    piece = env.scene[object_name]
    obj = piece.data.root_pos_w - env.scene.env_origins
    source = _source_target_world(env, "source").unsqueeze(0).expand_as(obj)
    piece_ready = _piece_on_square(
        obj, piece.data.root_quat_w, source, piece_source_xy_threshold, piece_source_z_threshold, piece_max_tilt_deg
    )
    grip = _gripper_joint_mean(env, gripper_joint_pattern)
    open_grip = grip >= gripper_open_threshold
    target = env._chess_pregrasp_target_pose
    valid = env._chess_pregrasp_target_valid
    external = env._chess_pregrasp_target_external

    # Generation installs an external transformed demonstration target before
    # this stage runs. During annotated recording there is no such target, so
    # use stricter proximity and stability thresholds for capturing the
    # operator's final open-gripper pose.
    piece_distance_limit = torch.where(
        external,
        torch.full_like(grip, piece_dist_threshold),
        torch.full_like(grip, recording_piece_dist_threshold),
    )
    position_limit = torch.where(
        external,
        torch.full_like(grip, position_tolerance),
        torch.full_like(grip, recording_position_tolerance),
    )
    orientation_limit = torch.where(
        external,
        torch.full_like(grip, orientation_tolerance),
        torch.full_like(grip, recording_orientation_tolerance),
    )
    position_step_limit = torch.where(
        external,
        torch.full_like(grip, max_position_step),
        torch.full_like(grip, recording_max_position_step),
    )
    orientation_step_limit = torch.where(
        external,
        torch.full_like(grip, max_orientation_step),
        torch.full_like(grip, recording_max_orientation_step),
    )
    required_stable_frames = torch.where(
        external,
        torch.full_like(env._chess_pregrasp_hold_counter, stable_frames),
        torch.full_like(env._chess_pregrasp_hold_counter, recording_stable_frames),
    )

    near_piece = torch.linalg.vector_norm(eef - obj, dim=-1) <= piece_distance_limit
    pending = _is_stage_pending(env, "pregrasp_align")
    candidate_ready = pending & piece_ready & open_grip & near_piece

    position_error, orientation_error = _pose_error(pose, target)
    refresh_target = candidate_ready & ~external & (
        ~valid | (position_error > position_limit) | (orientation_error > orientation_limit)
    )
    target[refresh_target] = pose[refresh_target]
    valid[refresh_target] = True
    env._chess_pregrasp_hold_counter[refresh_target] = 0
    position_error, orientation_error = _pose_error(pose, target)

    previous_pose = env._chess_pregrasp_prev_pose
    position_step, orientation_step = _pose_error(pose, previous_pose)
    episode_step = env.episode_length_buf.to(device=env.device, dtype=torch.long)
    last_counted = env._chess_pregrasp_last_counted_step
    new_step = last_counted != episode_step
    has_previous = last_counted >= 0
    stable = (
        candidate_ready
        & valid
        & (position_error <= position_limit)
        & (orientation_error <= orientation_limit)
        & has_previous
        & (position_step <= position_step_limit)
        & (orientation_step <= orientation_step_limit)
    )
    counter = env._chess_pregrasp_hold_counter
    counter[new_step & stable] += 1
    counter[new_step & ~stable] = 0
    previous_pose[new_step] = pose[new_step]
    last_counted[new_step] = episode_step[new_step]
    done = (counter >= required_stable_frames).float().unsqueeze(-1)
    _debug_print_subtask(
        env, signal_name, "pregrasp_align", float(position_error[0].item()), float(position_limit[0].item()), unit="m"
    )

    failed = ~piece_ready | (valid & ~open_grip)
    _mark_chess_failed(env, pending & failed)
    return _subtask_gate(env, "pregrasp_align", done, max_steps=max_steps)


def grasp_done(
    env: ManagerBasedRLEnv,
    eef_link: str,
    object_name: str | None = None,
    gripper_joint_pattern: str = "gripper",
    gripper_closed_threshold: float = 0.2,
    gripper_stability_tolerance: float = 0.005,
    stable_frames: int = 5,
    position_tolerance: float = 0.008,
    orientation_tolerance: float = 0.08,
    max_position_step: float = 0.0015,
    max_orientation_step: float = 0.03,
    max_position_departure: float = 0.02,
    max_orientation_departure: float = 0.20,
    piece_dist_threshold: float = 0.08,
    piece_source_xy_threshold: float = 0.08,
    piece_source_z_threshold: float = 0.06,
    piece_max_tilt_deg: float = 30.0,
    max_steps: int = 300,
    signal_name: str | None = None,
) -> torch.Tensor:
    """Hold the demonstrated grasp pose until the closed gripper is stable."""
    if stable_frames < 1:
        raise ValueError("stable_frames must be at least one")

    _ensure_pickup_state(env)
    pose = _eef_pose_robot_root(env, eef_link)
    target = env._chess_pregrasp_target_pose
    position_error, orientation_error = _pose_error(pose, target)
    target_valid = env._chess_pregrasp_target_valid

    if object_name is None:
        object_name = _get_active_piece_name(env)
    piece = env.scene[object_name]
    obj = piece.data.root_pos_w - env.scene.env_origins
    eef = _eef_position(env, eef_link)
    source = _source_target_world(env, "source").unsqueeze(0).expand_as(obj)
    piece_ready = _piece_on_square(
        obj, piece.data.root_quat_w, source, piece_source_xy_threshold, piece_source_z_threshold, piece_max_tilt_deg
    )
    near_piece = torch.linalg.vector_norm(eef - obj, dim=-1) <= piece_dist_threshold

    grip = _gripper_joint_mean(env, gripper_joint_pattern)
    closed = grip <= gripper_closed_threshold
    episode_step = env.episode_length_buf.to(device=env.device, dtype=torch.long)
    last_counted = env._chess_grasp_last_counted_step
    new_step = last_counted != episode_step
    has_previous = last_counted >= 0
    grip_stable = torch.abs(grip - env._chess_grasp_prev_grip) <= gripper_stability_tolerance
    grip_stable &= has_previous
    position_step, orientation_step = _pose_error(pose, env._chess_grasp_prev_pose)
    pose_stable = (
        has_previous
        & (position_step <= max_position_step)
        & (orientation_step <= max_orientation_step)
    )
    at_grasp_pose = (
        target_valid
        & (position_error <= position_tolerance)
        & (orientation_error <= orientation_tolerance)
    )
    stable = closed & grip_stable & pose_stable & at_grasp_pose & near_piece & piece_ready
    counter = env._chess_grasp_hold_counter
    counter[new_step & stable] += 1
    counter[new_step & ~stable] = 0
    env._chess_grasp_prev_grip[new_step] = grip[new_step]
    env._chess_grasp_prev_pose[new_step] = pose[new_step]
    last_counted[new_step] = episode_step[new_step]
    done = (counter >= stable_frames).float().unsqueeze(-1)
    _debug_print_subtask(
        env, signal_name, "grasp", float(grip[0].item()), float(gripper_closed_threshold), unit="rad"
    )

    pending = _is_stage_pending(env, "grasp")
    departed = (position_error > max_position_departure) | (orientation_error > max_orientation_departure)
    failed = ~piece_ready | (closed & (~near_piece | departed | ~target_valid))
    _mark_chess_failed(env, pending & failed)
    return _subtask_gate(env, "grasp", done, max_steps=max_steps)


def grasp_object_done(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible function alias for the renamed grasp predicate."""
    return grasp_done(*args, **kwargs)


def lift_object_done(
    env: ManagerBasedRLEnv,
    eef_link: str,
    object_name: str | None = None,
    source_dist_threshold: float = 0.08,
    lift_z_offset: float = 0.08,
    gripper_joint_pattern: str = "gripper",
    gripper_closed_threshold: float = 0.110,
    gripper_open_threshold: float = 0.5,
    gripper_lost_dist_threshold: float = 0.15,
    piece_drop_z_threshold: float = 0.03,
    piece_max_tilt_deg: float = 45.0,
    max_steps: int = 300,
    signal_name: str | None = None,
) -> torch.Tensor:
    """Latch a lift; dropping, releasing, tipping, or translating before lifting fails."""
    if object_name is None:
        object_name = _get_active_piece_name(env)
    piece = env.scene[object_name]
    obj = piece.data.root_pos_w - env.scene.env_origins
    source = _source_target_world(env, "source").unsqueeze(0).expand_as(obj)
    source_xy_distance = torch.linalg.vector_norm(obj[:, :2] - source[:, :2], dim=-1)
    grip = _gripper_joint_mean(env, gripper_joint_pattern)
    lifted = obj[:, 2] >= (source[:, 2] + lift_z_offset)
    near_source = source_xy_distance <= source_dist_threshold
    closed = grip <= gripper_closed_threshold
    done = (lifted & near_source & closed).float().unsqueeze(-1)
    _debug_print_subtask(
        env,
        signal_name,
        "lift_object",
        float(obj[0, 2].item() - source[0, 2].item()),
        float(lift_z_offset),
        unit="m",
    )

    eef = _eef_position(env, eef_link)
    lost_grip = torch.norm(eef - obj, dim=-1) > gripper_lost_dist_threshold
    released = grip >= gripper_open_threshold
    fell = obj[:, 2] < (source[:, 2] - piece_drop_z_threshold)
    translated_early = source_xy_distance > source_dist_threshold
    tipped = ~_piece_upright(piece.data.root_quat_w, piece_max_tilt_deg)
    failed = lost_grip | released | fell | translated_early | tipped
    pending = _is_stage_pending(env, "lift_object")
    _mark_chess_failed(env, pending & failed)

    latched = _subtask_gate(env, "lift_object", done, max_steps=max_steps)
    completed_env_ids = torch.nonzero(pending & done.squeeze(-1).bool(), as_tuple=False).flatten()
    if completed_env_ids.numel() > 0:
        clear_pickup_state(env, completed_env_ids)
    return latched


def move_over_destination_done(
    env: ManagerBasedRLEnv,
    eef_link: str,
    object_name: str | None = None,
    dist_threshold: float = 0.05,
    gripper_joint_pattern: str = "gripper",
    gripper_open_threshold: float = 0.5,
    gripper_lost_dist_threshold: float = 0.15,
    piece_drop_z_threshold: float = 0.05,
    piece_drop_xy_radius: float = 0.5,
    min_transport_height: float = 0.04,
    target_xy_threshold: float = 0.08,
    max_steps: int = 300,
    signal_name: str | None = None,
) -> torch.Tensor:
    """Latch transport; losing or prematurely setting down the piece fails."""
    eef = _eef_position(env, eef_link)
    source = _source_target_world(env, "source").unsqueeze(0).expand_as(eef)
    target = _source_target_world(env, "target").unsqueeze(0).expand_as(eef)

    dist_to_target = torch.norm(eef - target, dim=-1)
    dist_from_source = torch.norm(target - source, dim=-1)
    near_target = dist_to_target <= dist_threshold
    valid_move = dist_from_source > 0.0
    done = (near_target & valid_move).float().unsqueeze(-1)
    _debug_print_subtask(
        env, signal_name, "move_over_destination", float(dist_to_target[0].item()), float(dist_threshold), unit="m"
    )

    if object_name is None:
        object_name = _get_active_piece_name(env)
    obj = env.scene[object_name].data.root_pos_w - env.scene.env_origins
    is_pending = _is_stage_pending(env, "move_over_destination")

    grip = _gripper_joint_mean(env, gripper_joint_pattern)
    lost_grip = torch.norm(eef - obj, dim=-1) > gripper_lost_dist_threshold
    released = grip >= gripper_open_threshold
    board_z = torch.minimum(source[:, 2], target[:, 2])
    fell_z = obj[:, 2] < (board_z - piece_drop_z_threshold)
    fell_xy = torch.minimum(
        torch.norm(obj[:, :2] - source[:, :2], dim=-1), torch.norm(obj[:, :2] - target[:, :2], dim=-1)
    ) > piece_drop_xy_radius
    target_xy_distance = torch.linalg.vector_norm(obj[:, :2] - target[:, :2], dim=-1)
    landed_away_from_target = (obj[:, 2] <= board_z + min_transport_height) & (
        target_xy_distance > target_xy_threshold
    )
    failed = lost_grip | released | fell_z | fell_xy | landed_away_from_target
    _mark_chess_failed(env, is_pending & failed)

    return _subtask_gate(env, "move_over_destination", done, max_steps=max_steps)


def place_object_done(
    env: ManagerBasedRLEnv,
    eef_link: str,
    object_name: str | None = None,
    target_dist_threshold: float = 0.05,
    gripper_joint_pattern: str = "gripper",
    gripper_closed_threshold: float = 0.187,
    upright_angle_deg_min: float = -75.0,
    upright_angle_deg_max: float = 75.0,
    gripper_lost_dist_threshold: float = 0.15,
    release_xy_threshold: float = 0.05,
    release_z_threshold: float = 0.05,
    piece_drop_z_threshold: float = 0.05,
    piece_drop_xy_radius: float = 0.5,
    max_steps: int = 300,
    signal_name: str | None = None,
) -> torch.Tensor:
    """Latch placement; dropping or releasing outside the green square fails."""
    if object_name is None:
        object_name = _get_active_piece_name(env)

    piece = env.scene[object_name]
    obj = piece.data.root_pos_w - env.scene.env_origins
    target = _source_target_world(env, "target").unsqueeze(0).expand_as(obj)
    max_tilt_deg = max(abs(upright_angle_deg_min), abs(upright_angle_deg_max))
    upright = _piece_upright(piece.data.root_quat_w, max_tilt_deg)

    dist = torch.norm(obj - target, dim=-1)
    grip = _gripper_joint_mean(env, gripper_joint_pattern)
    near = dist <= target_dist_threshold
    opened = grip >= gripper_closed_threshold
    done = (near & opened & upright).float().unsqueeze(-1)
    _debug_print_subtask(
        env, signal_name, "place_object", float(dist[0].item()), float(target_dist_threshold), unit="m"
    )

    eef = _eef_position(env, eef_link)
    is_pending = _is_stage_pending(env, "place_object")
    valid_release = _piece_on_square(
        obj, piece.data.root_quat_w, target, release_xy_threshold, release_z_threshold, max_tilt_deg
    )
    bad_release = opened & ~valid_release
    lost_grip = ~opened & (torch.norm(eef - obj, dim=-1) > gripper_lost_dist_threshold)
    source = _source_target_world(env, "source").unsqueeze(0).expand_as(obj)
    board_z = torch.minimum(source[:, 2], target[:, 2])
    fell_z = obj[:, 2] < (board_z - piece_drop_z_threshold)
    fell_xy = torch.minimum(
        torch.norm(obj[:, :2] - source[:, :2], dim=-1), torch.norm(obj[:, :2] - target[:, :2], dim=-1)
    ) > piece_drop_xy_radius
    _mark_chess_failed(env, is_pending & (bad_release | lost_grip | fell_z | fell_xy))

    return _subtask_gate(env, "place_object", done, max_steps=max_steps)


def lift_after_place_done(
    env: ManagerBasedRLEnv,
    eef_link: str = "gripper_frame_link",
    object_name: str | None = None,
    lift_z_offset: float = 0.05,
    max_xy_drift: float = 0.02,
    gripper_joint_pattern: str = "gripper",
    gripper_open_threshold: float = 0.2,
    target_xy_threshold: float = 0.02,
    target_z_threshold: float = 0.02,
    piece_max_tilt_deg: float = 75.0,
    max_steps: int = 300,
    signal_name: str | None = None,
) -> torch.Tensor:
    """Latch a short vertical retreat after releasing the piece.

    The end-effector pose is captured when placement advances into this stage.
    Completion requires moving upward by ``lift_z_offset`` without translating
    farther than ``max_xy_drift`` in the board plane. The released piece must
    remain valid on its destination square throughout the retreat.
    """
    if lift_z_offset <= 0.0:
        raise ValueError("lift_z_offset must be positive.")
    if max_xy_drift < 0.0:
        raise ValueError("max_xy_drift must be nonnegative.")

    eef = _eef_position(env, eef_link)
    if not hasattr(env, "_chess_post_place_lift_start"):
        env._chess_post_place_lift_start = torch.zeros_like(eef)
    if not hasattr(env, "_chess_post_place_lift_initialized"):
        env._chess_post_place_lift_initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    start = env._chess_post_place_lift_start
    initialized = env._chess_post_place_lift_initialized
    is_pending = _is_stage_pending(env, "lift_after_place")
    capture = is_pending & ~initialized
    start[capture] = eef[capture]
    initialized[capture] = True

    xy_drift = torch.linalg.vector_norm(eef[:, :2] - start[:, :2], dim=-1)
    vertical_lift = eef[:, 2] - start[:, 2]
    grip = _gripper_joint_mean(env, gripper_joint_pattern)
    opened = grip >= gripper_open_threshold

    if object_name is None:
        object_name = _get_active_piece_name(env)
    piece = env.scene[object_name]
    obj = piece.data.root_pos_w - env.scene.env_origins
    target = _source_target_world(env, "target").unsqueeze(0).expand_as(obj)
    piece_still_placed = _piece_on_square(
        obj,
        piece.data.root_quat_w,
        target,
        target_xy_threshold,
        target_z_threshold,
        piece_max_tilt_deg,
    )

    straight = xy_drift <= max_xy_drift
    lifted = vertical_lift >= lift_z_offset
    done = (initialized & straight & lifted & opened & piece_still_placed).float().unsqueeze(-1)
    _debug_print_subtask(
        env,
        signal_name,
        "lift_after_place",
        float(vertical_lift[0].item()),
        float(lift_z_offset),
        unit="m",
    )

    invalid_retreat = ~straight | ~opened | ~piece_still_placed
    _mark_chess_failed(env, is_pending & initialized & invalid_retreat)
    return _subtask_gate(env, "lift_after_place", done, max_steps=max_steps)


def return_home_done(
    env: ManagerBasedRLEnv,
    object_name: str | None = None,
    eef_link: str = "gripper_frame_link",
    home_pos: tuple[float, float, float] = (0.05, -0.12, 0.0135),
    home_threshold: tuple[float, float, float] = (0.04, 0.04, 0.03),
    min_home_steps: int = 5,
    max_home_step_m: float = 0.001,
    gripper_joint_pattern: str = "gripper",
    gripper_closed_threshold: float = -0.12,
    target_xy_threshold: float = 0.02,
    target_z_threshold: float = 0.02,
    return_start_max_tilt_deg: float = 75.0,
    piece_max_tilt_deg: float = 5.0,
    max_steps: int = 300,
    signal_name: str | None = None,
    require_current: bool = False,
) -> torch.Tensor:
    """Latch return-home after the Cartesian end effector has settled at home.

    While the robot is returning, the piece must remain on the destination
    within ``return_start_max_tilt_deg``. The stricter
    ``piece_max_tilt_deg`` check is applied only once the end effector has
    remained inside ``home_threshold`` while moving at most ``max_home_step_m`` per step for
    ``min_home_steps`` consecutive return-home steps and the gripper is closed.
    """
    if min_home_steps < 1:
        raise ValueError("min_home_steps must be at least one.")

    eef = _eef_position_robot_root(env, eef_link)
    home = torch.tensor(home_pos, dtype=eef.dtype, device=eef.device).unsqueeze(0)
    threshold = torch.tensor(home_threshold, dtype=eef.dtype, device=eef.device).unsqueeze(0)
    delta = torch.abs(eef - home)
    near_home = (delta <= threshold).all(dim=-1)
    grip = _gripper_joint_mean(env, gripper_joint_pattern)
    closed = grip <= gripper_closed_threshold
    stage = getattr(env, "_chess_subtask_stage", None)
    returning_home = (
        stage >= _SUBTASK_ORDER.index("return_home")
        if stage is not None
        else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )

    if not hasattr(env, "_chess_home_hold_counter"):
        env._chess_home_hold_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_chess_home_last_counted_step"):
        env._chess_home_last_counted_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "_chess_home_prev_eef"):
        env._chess_home_prev_eef = torch.zeros_like(eef)
    home_hold_counter = env._chess_home_hold_counter
    last_counted_step = env._chess_home_last_counted_step
    previous_eef = env._chess_home_prev_eef
    episode_step = env.episode_length_buf.to(device=env.device, dtype=torch.long)
    new_step = last_counted_step != episode_step
    settled = (eef - previous_eef).abs().amax(dim=-1) <= max_home_step_m
    settled &= last_counted_step >= 0
    at_home = returning_home & near_home & closed & settled
    home_hold_counter[new_step & at_home] += 1
    home_hold_counter[new_step & ~at_home] = 0
    previous_eef[new_step] = eef[new_step]
    last_counted_step[new_step] = episode_step[new_step]
    motion_complete = home_hold_counter >= int(min_home_steps)
    _debug_print_subtask(
        env,
        signal_name,
        "return_home",
        float(delta.max().item()),
        float(threshold.max().item()),
        unit="m",
    )

    if object_name is None:
        object_name = _get_active_piece_name(env)
    piece = env.scene[object_name]
    obj = piece.data.root_pos_w - env.scene.env_origins
    target = _source_target_world(env, "target").unsqueeze(0).expand_as(obj)
    placement_intact_during_return = _piece_on_square(
        obj,
        piece.data.root_quat_w,
        target,
        target_xy_threshold,
        target_z_threshold,
        return_start_max_tilt_deg,
    )
    placement_valid_at_home = _piece_on_square(
        obj, piece.data.root_quat_w, target, target_xy_threshold, target_z_threshold, piece_max_tilt_deg
    )

    # Do not apply the strict final-angle check on the frame that placement
    # advances into return-home. It becomes mandatory only after the robot has
    # completed the return motion.
    failed = ~placement_intact_during_return | (motion_complete & ~placement_valid_at_home)
    _mark_chess_failed(env, _is_stage_pending(env, "return_home") & failed)

    done = (motion_complete & placement_valid_at_home).float().unsqueeze(-1)
    latched = _subtask_gate(env, "return_home", done, max_steps=max_steps)
    return done * latched if require_current else latched
