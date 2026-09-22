# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-safe joint actions for SO-101 chess teleoperation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from .board import square_surface_position

class SlewRateLimitedJointPositionAction(JointPositionAction):
    """Absolute joint targets with a physical velocity limit.

    The optional board guard uses the end-effector Jacobian to reject only
    downward arm commands while a grasped piece is at the board.  Upward
    commands remain available, so lifting immediately after a grasp still
    works.  The gripper action uses the same class without the board guard.
    """

    cfg: "SlewRateLimitedJointPositionActionCfg"

    def __init__(self, cfg: "SlewRateLimitedJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._previous_target = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._max_delta = float(cfg.max_velocity) * float(env.step_dt)
        if isinstance(self._joint_ids, slice):
            joint_ids = list(range(self._asset.num_joints))[self._joint_ids]
        else:
            joint_ids = self._joint_ids
        self._joint_ids_tensor = torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)

        self._eef_jacobian_index: int | None = None
        self._eef_body_index: int | None = None
        self._gripper_joint_index: int | None = None
        self._board_surface_heights: torch.Tensor | None = None
        if cfg.block_downward_near_board:
            body_index = self._asset.data.body_names.index(cfg.eef_body_name)
            self._eef_body_index = body_index
            self._eef_jacobian_index = body_index - 1 if self._asset.is_fixed_base else body_index
            gripper_ids, _ = self._asset.find_joints(cfg.gripper_joint_name)
            if len(gripper_ids) != 1:
                raise ValueError(
                    f"Expected one joint named {cfg.gripper_joint_name!r}, found {len(gripper_ids)}."
                )
            self._gripper_joint_index = gripper_ids[0]

    def _surface_heights(self) -> torch.Tensor:
        if self._board_surface_heights is None:
            heights = [
                square_surface_position(self._env.sim.stage, env_id, 0, 0)[2]
                for env_id in range(self.num_envs)
            ]
            self._board_surface_heights = torch.tensor(
                heights,
                dtype=self._asset.data.root_pos_w.dtype,
                device=self.device,
            )
        return self._board_surface_heights

    def _block_downward_board_contact(self, targets: torch.Tensor) -> torch.Tensor:
        if self._eef_jacobian_index is None or self._gripper_joint_index is None:
            return targets

        piece_name = getattr(self._env, "active_piece_name", None)
        if not isinstance(piece_name, str) or piece_name not in self._env.scene.keys():
            return targets

        current = self._asset.data.joint_pos[:, self._joint_ids]
        delta = targets - current
        jacobians = self._asset.root_physx_view.get_jacobians()
        eef_vertical_jacobian = jacobians[
            :, self._eef_jacobian_index, 2, self._joint_ids_tensor
        ]
        requested_vertical_motion = torch.sum(eef_vertical_jacobian * delta, dim=-1)

        piece = self._env.scene[piece_name]
        eef_position = self._asset.data.body_pos_w[:, self._eef_body_index]
        near_piece = torch.linalg.vector_norm(eef_position - piece.data.root_pos_w, dim=-1) <= float(
            self.cfg.grasp_proximity_threshold
        )
        near_board = piece.data.root_pos_w[:, 2] <= (
            self._surface_heights() + float(self.cfg.board_contact_clearance)
        )
        gripper_closed = (
            self._asset.data.joint_pos[:, self._gripper_joint_index]
            <= float(self.cfg.gripper_closed_threshold)
        )
        blocked = near_piece & near_board & gripper_closed & (requested_vertical_motion < 0.0)
        return torch.where(blocked.unsqueeze(-1), current, targets)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        delta = torch.clamp(
            self._processed_actions - self._previous_target,
            min=-self._max_delta,
            max=self._max_delta,
        )
        targets = self._previous_target + delta
        targets = self._block_downward_board_contact(targets)
        self._processed_actions = targets
        self._previous_target[:] = targets

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        current = self._asset.data.joint_pos[:, self._joint_ids]
        if env_ids is None:
            self._previous_target[:] = current
            return
        self._previous_target[env_ids] = current[env_ids]


@configclass
class SlewRateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`SlewRateLimitedJointPositionAction`."""

    class_type: type = SlewRateLimitedJointPositionAction

    max_velocity: float = 1.0
    """Maximum target slew rate in radians per second."""

    block_downward_near_board: bool = False
    """Reject downward end-effector commands while a held piece touches the board."""

    eef_body_name: str = "gripper_frame_link"
    gripper_joint_name: str = "gripper"
    gripper_closed_threshold: float = 0.30
    grasp_proximity_threshold: float = 0.08
    board_contact_clearance: float = 0.002
