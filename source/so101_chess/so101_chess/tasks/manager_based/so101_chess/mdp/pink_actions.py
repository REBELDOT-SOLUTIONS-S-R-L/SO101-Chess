# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pink IK action support for the SO-101 chess task."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from pink import solve_ik

from isaaclab.controllers.pink_ik import PinkIKController
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from .board import square_surface_position


ARM_TARGET_MAX_VELOCITY = 2.0
GRIPPER_CLOSE_TARGET = -0.17
GRIPPER_OPEN_LIMIT = 1.74533
GRIPPER_CLOSED_THRESHOLD = 0.30
BOARD_CONTACT_CLEARANCE = 0.002
GRIPPER_HOLDING_DISTANCE = 0.10
GRIPPER_EMPTY_CLOSED_MARGIN = 0.02
HOME_EEF_POS = (0.05, -0.12, 0.0135)
HOME_EEF_THRESHOLD = (0.04, 0.04, 0.03)


class QuadprogPinkIKController(PinkIKController):
    """Pink controller using the QP backend compatible with this environment.

    Isaac Lab currently hardcodes DAQP, but qpsolvers 4.12 passes a keyword
    that the installed DAQP 0.7.2 binding does not accept. Quadprog solves the
    same Pink QP and is already available in the project environment.
    """

    def compute(self, curr_joint_pos: np.ndarray, dt: float) -> torch.Tensor:
        """Compute one joint-position target from the current configuration."""
        curr_controlled_joint_pos = [curr_joint_pos[index] for index in self.controlled_joint_indices]
        joint_positions_pink = curr_joint_pos[self.isaac_lab_to_pink_ordering]
        self.pink_configuration.update(joint_positions_pink)

        try:
            velocity = solve_ik(
                self.pink_configuration,
                self.cfg.variable_input_tasks + self.cfg.fixed_input_tasks,
                dt,
                solver="quadprog",
                safety_break=self.cfg.fail_on_joint_limit_violation,
            )
            joint_angle_changes = velocity * dt
        except Exception as error:
            if self.cfg.show_ik_warnings:
                print(f"Warning: Pink/quadprog solve failed; holding current joints. Error: {error}")
            return torch.tensor(curr_controlled_joint_pos, device=self.device, dtype=torch.float32)

        joint_angle_changes_isaac = torch.tensor(
            joint_angle_changes[self.pink_to_isaac_lab_controlled_ordering],
            device=self.device,
            dtype=torch.float32,
        )
        return joint_angle_changes_isaac + torch.tensor(
            curr_controlled_joint_pos,
            device=self.device,
            dtype=torch.float32,
        )


class So101PinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Pink action whose Cartesian command is expressed in robot-root space.

    The task's Mimic API and recorded ``actions/pose`` values use the robot
    root as their reference frame. Isaac Lab's stock Pink action instead
    interprets incoming poses in the environment frame and transforms them to
    the base link. The calibrated SO-101 URDF already uses the robot root as
    its base frame, so this action sends those poses to Pink directly.
    """

    def _initialize_ik_controllers(self) -> None:
        if self._env.num_envs < 1:
            raise ValueError("Pink requires at least one environment.")
        self._ik_controllers = [
            QuadprogPinkIKController(
                cfg=self.cfg.controller.copy(),
                robot_cfg=getattr(self._env.scene.cfg, self.cfg.controller.articulation_name),
                device=self.device,
                controlled_joint_indices=self._isaaclab_controlled_joint_ids,
            )
            for _ in range(self._env.num_envs)
        ]

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self.hand_joint_dim != 1 or self._hand_joint_names != ["gripper"]:
            raise ValueError("The SO-101 Pink action expects one direct 'gripper' joint target.")
        if self.action_dim != 8:
            raise ValueError(f"Expected a seven-value pose and one gripper target, got {self.action_dim} values.")

        # Pink is evaluated at each physics step, but a one-physics-step target
        # lead is too small for the SO-101's position-drive gains. Use one
        # environment control period as the solver horizon.
        self._sim_dt = env.step_dt
        self._physics_dt = env.sim.get_physics_dt()

        self._arm_joint_ids_tensor = torch.as_tensor(
            self._isaaclab_controlled_joint_ids, dtype=torch.long, device=self.device
        )
        self._previous_arm_target = self._asset.data.default_joint_pos[
            :, self._isaaclab_controlled_joint_ids
        ].clone()
        self._target_hand_joint_positions = self._asset.data.default_joint_pos[
            :, self._hand_joint_ids
        ].clone()
        self._home_target_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        body_index = self._asset.data.body_names.index("gripper_frame_link")
        self._eef_body_index = body_index
        self._eef_jacobian_index = body_index - 1 if self._asset.is_fixed_base else body_index
        self._gripper_joint_index = self._hand_joint_ids[0]
        self._board_surface_heights: torch.Tensor | None = None

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
        """Hold the arm if its target would push a grasped piece into the board."""
        stage = getattr(self._env, "_chess_subtask_stage", None)
        piece_name = getattr(self._env, "active_piece_name", None)
        if not isinstance(piece_name, str) or piece_name not in self._env.scene.keys():
            return targets

        current = self._asset.data.joint_pos[:, self._isaaclab_controlled_joint_ids]
        delta = targets - current
        jacobians = self._asset.root_physx_view.get_jacobians()
        vertical_jacobian = jacobians[
            :, self._eef_jacobian_index, 2, self._arm_joint_ids_tensor
        ]
        requested_vertical_motion = torch.sum(vertical_jacobian * delta, dim=-1)

        piece = self._env.scene[piece_name]
        near_board = piece.data.root_pos_w[:, 2] <= (
            self._surface_heights() + BOARD_CONTACT_CLEARANCE
        )
        gripper_closed = (
            self._asset.data.joint_pos[:, self._gripper_joint_index]
            <= GRIPPER_CLOSED_THRESHOLD
        )
        gripper_obstructed = self._asset.data.joint_pos[:, self._gripper_joint_index] > (
            GRIPPER_CLOSE_TARGET + GRIPPER_EMPTY_CLOSED_MARGIN
        )
        piece_near_gripper = torch.linalg.vector_norm(
            piece.data.root_pos_w - self._asset.data.body_pos_w[:, self._eef_body_index], dim=-1
        ) <= GRIPPER_HOLDING_DISTANCE
        blocked = (
            near_board
            & gripper_closed
            & gripper_obstructed
            & piece_near_gripper
            & (requested_vertical_motion < 0.0)
        )
        if stage is not None:
            # Once place_object (stage 5) has latched, the nearby resting
            # piece is no longer held by the gripper.
            blocked &= stage < 6
        return torch.where(blocked.unsqueeze(-1), current, targets)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Set the robot-root EEF target and direct gripper command."""
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"Expected Pink actions shaped {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}."
            )

        self._raw_actions[:] = actions
        self._target_hand_joint_positions = actions[:, -self.hand_joint_dim :]

        controlled_frame_poses = self._extract_controlled_frame_poses(actions)
        positions = controlled_frame_poses[..., :3, 3]
        rotation_matrices = controlled_frame_poses[..., :3, :3]
        home_pos = torch.tensor(HOME_EEF_POS, dtype=positions.dtype, device=positions.device)
        home_threshold = torch.tensor(HOME_EEF_THRESHOLD, dtype=positions.dtype, device=positions.device)
        home_target_per_frame = (torch.abs(positions - home_pos) <= home_threshold).all(dim=-1)
        self._home_target_mask = home_target_per_frame.reshape(self.num_envs, -1).all(dim=-1)
        self._set_task_targets((positions, rotation_matrices))

    def apply_actions(self) -> None:
        """Apply the guarded arm target and the recorded direct gripper target."""
        desired_arm = self._compute_ik_solutions()
        default_arm = self._asset.data.default_joint_pos[:, self._isaaclab_controlled_joint_ids]
        desired_arm = torch.where(self._home_target_mask.unsqueeze(-1), default_arm, desired_arm)
        arm_max_delta = ARM_TARGET_MAX_VELOCITY * self._physics_dt
        arm_delta = torch.clamp(
            desired_arm - self._previous_arm_target,
            min=-arm_max_delta,
            max=arm_max_delta,
        )
        arm_target = self._previous_arm_target + arm_delta
        arm_target = self._block_downward_board_contact(arm_target)

        desired_hand = torch.clamp(
            self._target_hand_joint_positions,
            min=GRIPPER_CLOSE_TARGET,
            max=GRIPPER_OPEN_LIMIT,
        )
        self._previous_arm_target[:] = arm_target
        self._processed_actions = torch.cat((arm_target, desired_hand), dim=1)

        if self.cfg.enable_gravity_compensation:
            self._apply_gravity_compensation()
        self._asset.set_joint_position_target(self._processed_actions, self._controlled_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        current_arm = self._asset.data.joint_pos[:, self._isaaclab_controlled_joint_ids]
        current_hand = self._asset.data.joint_pos[:, self._hand_joint_ids]
        if env_ids is None:
            self._previous_arm_target[:] = current_arm
            self._target_hand_joint_positions[:] = current_hand
            return
        self._previous_arm_target[env_ids] = current_arm[env_ids]
        self._target_hand_joint_positions[env_ids] = current_hand[env_ids]
