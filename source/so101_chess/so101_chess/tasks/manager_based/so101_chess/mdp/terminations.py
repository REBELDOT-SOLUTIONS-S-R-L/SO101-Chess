# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific termination functions for so101_chess.

Add custom termination conditions here. They will be available as `mdp.<func_name>`
in your task config's TerminationsCfg.

Base terminations (time_out, etc.) are already available from isaaclab.envs.mdp.
Shared IL helpers (get_object_pos, to_tensor, etc.) are available from the
base_il_env mdp.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply

from .observations import (
    _get_active_piece_name,
    _source_target_world,
    get_chess_subtask_failed,
    return_home_done,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def chess_move_success(
    env: ManagerBasedRLEnv,
    object_name: str | None = None,
    target_xy_threshold: float = 0.02,
    target_z_threshold: float = 0.02,
    max_tilt_deg: float = 5.0,
    return_start_max_tilt_deg: float = 75.0,
    home_eef_link: str = "gripper_frame_link",
    home_pos: tuple[float, float, float] = (0.05, -0.12, 0.0135),
    home_threshold: tuple[float, float, float] = (0.04, 0.04, 0.03),
    min_home_steps: int = 5,
    max_home_step_m: float = 0.001,
    gripper_joint_pattern: str = "gripper",
    gripper_closed_threshold: float = 0.30,
    min_hold_steps: int = 5,
) -> torch.Tensor:
    """Return true after a correctly placed piece and completed return-home step.

    The target tolerance is smaller than the 3.33 cm center-to-center square
    spacing, so a piece on an adjacent square cannot count as successful.
    The piece must remain on the destination square (xy/z/upright all true)
    for at least ``min_hold_steps`` consecutive environment steps before
    success is reported.
    """
    if object_name is None:
        object_name = _get_active_piece_name(env)

    piece = env.scene[object_name]
    piece_position = piece.data.root_pos_w - env.scene.env_origins
    target_position = _source_target_world(env, "target").unsqueeze(0).expand_as(piece_position)

    xy_distance = torch.linalg.vector_norm(piece_position[:, :2] - target_position[:, :2], dim=-1)
    z_distance = torch.abs(piece_position[:, 2] - target_position[:, 2])

    local_up = torch.tensor(
        [0.0, 0.0, 1.0], device=piece.data.root_quat_w.device, dtype=piece.data.root_quat_w.dtype
    ).expand(piece.data.root_quat_w.shape[0], 3)
    piece_up = quat_apply(piece.data.root_quat_w, local_up)
    upright = piece_up[:, 2] >= math.cos(math.radians(max_tilt_deg))

    on_destination = (xy_distance <= target_xy_threshold) & (z_distance <= target_z_threshold) & upright

    # Persistent per-env counter of consecutive steps the piece has stayed on target.
    if not hasattr(env, "_chess_on_dest_hold_counter"):
        env._chess_on_dest_hold_counter = torch.zeros(
            env.num_envs, dtype=torch.long, device=on_destination.device
        )
    counter = env._chess_on_dest_hold_counter
    counter[on_destination] += 1
    counter[~on_destination] = 0
    held_long_enough = counter >= min_hold_steps

    returned_home = return_home_done(
        env,
        eef_link=home_eef_link,
        home_pos=home_pos,
        home_threshold=home_threshold,
        min_home_steps=min_home_steps,
        max_home_step_m=max_home_step_m,
        gripper_joint_pattern=gripper_joint_pattern,
        gripper_closed_threshold=gripper_closed_threshold,
        target_xy_threshold=target_xy_threshold,
        target_z_threshold=target_z_threshold,
        return_start_max_tilt_deg=return_start_max_tilt_deg,
        piece_max_tilt_deg=max_tilt_deg,
        require_current=True,
    ).squeeze(-1).bool()

    succeeded = held_long_enough & returned_home
    if not hasattr(env, "_chess_move_episode_succeeded"):
        env._chess_move_episode_succeeded = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._chess_move_episode_succeeded |= succeeded
    return succeeded


def reset_chess_hold_counter(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset the destination-hold counter for the given env ids on episode reset.

    Register this as an EventTermCfg with mode="reset" in your task config so
    the counter used by ``chess_move_success`` doesn't leak state across resets.
    """
    if hasattr(env, "_chess_on_dest_hold_counter"):
        env._chess_on_dest_hold_counter[env_ids] = 0


def chess_move_failed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return true (episode failed) once any subtask-specific fail condition has latched.

    Each subtask's done() function in observations.py owns its own fail checks
    (dropped piece, lost grip, stalled subtask) and ORs them into a shared
    per-env latch; this termination just reads that latch.
    """
    return get_chess_subtask_failed(env)
