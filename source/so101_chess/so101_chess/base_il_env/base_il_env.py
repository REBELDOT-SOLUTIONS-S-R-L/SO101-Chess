# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base IL environment class implementing the 4 mandatory ManagerBasedRLMimicEnv methods.

This class is designed so that you NEVER need to subclass it for new tasks.
All task-specific behavior is driven by config (BaseILEnvCfg subclasses).

The 4 mandatory methods use config attributes to know how to slice the action tensor:
    - cfg.eef_names: list of EEF names, e.g. ["left", "right"]
    - cfg.eef_action_slices: dict mapping eef_name -> (pos_start, quat_start, quat_end)
    - cfg.eef_gripper_slices: dict mapping eef_name -> (gripper_start, gripper_end)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from pxr import Usd, UsdGeom

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class BaseILEnv(ManagerBasedRLMimicEnv):
    """Shared IL environment for all tasks.

    Implements the 4 mandatory methods of ManagerBasedRLMimicEnv generically.
    Works for single-arm and dual-arm robots — the number of arms and action
    layout is configured entirely through cfg attributes.

    Required config attributes:
        eef_names (list[str]): Names of end-effectors, e.g. ["left", "right"].
        eef_action_slices (dict): Per-EEF action tensor layout.
            Each entry: eef_name -> {"pos": (start, end), "quat": (start, end)}
        eef_gripper_slices (dict): Per-EEF gripper tensor layout.
            Each entry: eef_name -> (start, end)
    """

    # ------------------------------------------------------------------
    # 1) get_robot_eef_pose
    # ------------------------------------------------------------------
    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end-effector pose as a 4x4 matrix.

        Reads ``{eef_name}_eef_pos`` and ``{eef_name}_eef_quat`` from the
        observation buffer. Observations must follow this naming convention
        (set via ObsTerm in your config).

        Args:
            eef_name: Name of the end effector (must match a key in cfg.eef_names).
            env_ids: Environment indices. If None, all envs are returned.

        Returns:
            Pose matrix of shape (len(env_ids), 4, 4).
        """
        if env_ids is None:
            env_ids = slice(None)

        # The generated observations store position relative to the environment
        # origin and orientation in the world frame. Mimic expects the pose in
        # the robot-root frame used by the absolute task-space controller.
        robot = self.scene["robot"]
        eef_pos_w = (
            self.obs_buf["policy"][f"{eef_name}_eef_pos"][env_ids]
            + self.scene.env_origins[env_ids]
        )
        eef_quat_w = self.obs_buf["policy"][f"{eef_name}_eef_quat"][env_ids]
        eef_pos, eef_quat = PoseUtils.subtract_frame_transforms(
            robot.data.root_pos_w[env_ids],
            robot.data.root_quat_w[env_ids],
            eef_pos_w,
            eef_quat_w,
        )
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    # ------------------------------------------------------------------
    # 2) target_eef_pose_to_action
    # ------------------------------------------------------------------
    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF poses + gripper actions into an env action tensor.

        Builds the action tensor in the environment action layout described by
        ``cfg.eef_action_slices`` and ``cfg.eef_gripper_slices``.

        Args:
            target_eef_pose_dict: Maps eef_name -> 4x4 target pose.
            gripper_action_dict: Maps eef_name -> gripper action tensor.
            action_noise_dict: Optional per-EEF noise scale.
            env_id: Environment index (unused for absolute actions, kept for API compliance).

        Returns:
            Flat action tensor compatible with env.step().
        """
        action_dim = 0
        for eef_name in self.cfg.eef_names:
            slices = self.cfg.eef_action_slices[eef_name]
            action_dim = max(action_dim, slices["pos"][1], slices["quat"][1])
            gripper_sel = self.cfg.eef_gripper_slices[eef_name]
            if isinstance(gripper_sel, tuple) and len(gripper_sel) == 2:
                action_dim = max(action_dim, gripper_sel[1])
            else:
                action_dim = max(action_dim, max(gripper_sel, default=-1) + 1)

        action = None

        for eef_name in self.cfg.eef_names:
            target_pose = target_eef_pose_dict[eef_name]
            target_pos, target_rot = PoseUtils.unmake_pose(target_pose)
            target_quat = PoseUtils.quat_from_matrix(target_rot)

            gripper_action = gripper_action_dict[eef_name]

            if action_noise_dict is not None and eef_name in action_noise_dict:
                noise_scale = action_noise_dict[eef_name]
                pos_noise = noise_scale * torch.randn_like(target_pos)
                quat_noise = noise_scale * torch.randn_like(target_quat)
                target_pos = target_pos + pos_noise
                target_quat = target_quat + quat_noise

            if action is None:
                action = torch.zeros(
                    (*target_pos.shape[:-1], action_dim),
                    dtype=target_pos.dtype,
                    device=target_pos.device,
                )

            slices = self.cfg.eef_action_slices[eef_name]
            pos_s, pos_e = slices["pos"]
            quat_s, quat_e = slices["quat"]
            action[..., pos_s:pos_e] = target_pos
            action[..., quat_s:quat_e] = target_quat

            # Waypoint noise belongs to the Cartesian arm command only. Copy
            # the demonstration's gripper command after applying pose noise so
            # the gripper value remains unchanged.
            gripper_sel = self.cfg.eef_gripper_slices[eef_name]
            if isinstance(gripper_sel, tuple) and len(gripper_sel) == 2:
                g_start, g_end = gripper_sel
                action[..., g_start:g_end] = gripper_action
            else:
                idx = torch.as_tensor(gripper_sel, dtype=torch.long, device=action.device)
                action[..., idx] = gripper_action

        if action is None:
            raise ValueError("Cannot build action because cfg.eef_names is empty.")

        return action

    # ------------------------------------------------------------------
    # 3) action_to_target_eef_pose
    # ------------------------------------------------------------------
    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert an action tensor back into per-EEF target poses.

        Uses ``cfg.eef_action_slices`` to know where pos and quat live
        inside the action tensor for each arm.

        Args:
            action: Action tensor of shape (num_envs, action_dim).

        Returns:
            Dict mapping eef_name -> 4x4 pose tensor of shape (num_envs, 4, 4).
        """
        if getattr(self.cfg, "task_type", None) == "so101leader":
            # Physical leader demonstrations contain six ordered joint targets,
            # not a Cartesian pose. Record the follower's measured EEF pose so
            # the resulting dataset still contains valid Mimic datagen fields.
            return {eef_name: self.get_robot_eef_pose(eef_name) for eef_name in self.cfg.eef_names}

        target_poses = {}

        for eef_name in self.cfg.eef_names:
            slices = self.cfg.eef_action_slices[eef_name]
            pos_s, pos_e = slices["pos"]
            quat_s, quat_e = slices["quat"]

            pos = action[:, pos_s:pos_e]
            quat = action[:, quat_s:quat_e]
            rot_mat = PoseUtils.matrix_from_quat(quat)
            target_poses[eef_name] = PoseUtils.make_pose(pos, rot_mat)

        return target_poses

    # ------------------------------------------------------------------
    # 4) actions_to_gripper_actions
    # ------------------------------------------------------------------
    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract per-EEF gripper actions from the full action tensor.

        ``cfg.eef_gripper_slices[eef_name]`` may be either:
          * ``(start, end)`` — contiguous slice (e.g. simple parallel grippers).
          * ``list[int]`` of column indices — for interleaved layouts like the
            Inspire-hand where left/right finger joints are not contiguous.

        Args:
            actions: Action tensor of shape (num_envs, num_steps, action_dim)
                     or (num_envs, action_dim).

        Returns:
            Dict mapping eef_name -> gripper action tensor.
        """
        if getattr(self.cfg, "task_type", None) == "so101leader":
            if len(self.cfg.eef_names) != 1:
                raise ValueError("SO-101 leader control requires exactly one configured end effector.")
            gripper_action = self.action_manager.get_term("gripper_action").processed_actions
            return {self.cfg.eef_names[0]: gripper_action}

        result = {}
        for eef_name in self.cfg.eef_names:
            sel = self.cfg.eef_gripper_slices[eef_name]
            if isinstance(sel, tuple) and len(sel) == 2:
                g_start, g_end = sel
                result[eef_name] = actions[..., g_start:g_end]
            else:
                idx = torch.as_tensor(sel, dtype=torch.long, device=actions.device)
                result[eef_name] = actions[..., idx]
        return result

    # ------------------------------------------------------------------
    # 5) get_object_poses
    # ------------------------------------------------------------------
    # The base class has a working implementation, but the Mimic recorder's
    # _require_mimic_methods check rejects methods whose qualname starts with
    # "ManagerBasedRLMimicEnv." — so we re-declare the same logic here.
    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return Mimic's chess-piece and destination poses in the robot-root frame.

        The chess scene keeps every piece as a rigid object and parks inactive
        pieces off-board. The manipulated piece is exposed under the stable
        ``active_piece`` name shared by every demonstration.

        The destination is a board square rather than a rigid scene object, so
        it is exposed as a virtual ``destination_square`` pose. This lets the
        destination-side Mimic subtasks jointly select demonstrations using the
        piece and target layout and transform their trajectories into the
        current target-square frame.
        """
        if env_ids is None:
            resolved_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        elif isinstance(env_ids, slice):
            resolved_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)[env_ids]
        else:
            resolved_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        active_piece_name = getattr(self, "active_piece_name", None)
        if not isinstance(active_piece_name, str):
            raise RuntimeError("The active chess piece has not been selected for this episode.")

        active_piece = self.scene.rigid_objects.get(active_piece_name)
        if active_piece is None:
            raise KeyError(f"Active chess piece '{active_piece_name}' is not a rigid object in the scene.")

        target_square = getattr(self, "active_target_square", None)
        if not isinstance(target_square, (tuple, list)) or len(target_square) != 2:
            raise RuntimeError("The destination chess square has not been selected for this episode.")
        target_row, target_column = int(target_square[0]), int(target_square[1])
        if not (0 <= target_row <= 7 and 0 <= target_column <= 7):
            raise ValueError(f"Invalid destination chess square: {target_square}")

        robot = self.scene["robot"]
        robot_pos_w = robot.data.root_pos_w[resolved_env_ids]
        robot_quat_w = robot.data.root_quat_w[resolved_env_ids]
        object_pos, object_quat = PoseUtils.subtract_frame_transforms(
            robot_pos_w,
            robot_quat_w,
            active_piece.data.root_pos_w[resolved_env_ids],
            active_piece.data.root_quat_w[resolved_env_ids],
        )

        bbox_cache = getattr(self, "_mimic_board_bbox_cache", None)
        if bbox_cache is None:
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            self._mimic_board_bbox_cache = bbox_cache

        position_cache = getattr(self, "_mimic_square_position_cache", None)
        if position_cache is None:
            position_cache = {}
            self._mimic_square_position_cache = position_cache

        destination_positions_w = []
        for env_id in resolved_env_ids.cpu().tolist():
            cache_key = (env_id, target_row, target_column)
            if cache_key not in position_cache:
                square_path = (
                    f"/World/envs/env_{env_id}/Scene/ChessBoard/"
                    f"Square_{target_row}_{target_column}"
                )
                square_prim = self.sim.stage.GetPrimAtPath(square_path)
                if not square_prim.IsValid():
                    raise RuntimeError(f"Destination square prim does not exist: {square_path}")

                bounds = bbox_cache.ComputeWorldBound(square_prim).ComputeAlignedRange()
                minimum = bounds.GetMin()
                maximum = bounds.GetMax()
                position_cache[cache_key] = (
                    (minimum[0] + maximum[0]) * 0.5,
                    (minimum[1] + maximum[1]) * 0.5,
                    maximum[2] + 0.001,
                )
            destination_positions_w.append(position_cache[cache_key])

        destination_pos_w = torch.tensor(
            destination_positions_w,
            dtype=robot_pos_w.dtype,
            device=self.device,
        )
        # Align the virtual frame with the robot root. Its pose therefore
        # carries target translation without introducing a synthetic rotation.
        destination_pos, destination_quat = PoseUtils.subtract_frame_transforms(
            robot_pos_w,
            robot_quat_w,
            destination_pos_w,
            robot_quat_w,
        )
        return {
            "active_piece": PoseUtils.make_pose(
                object_pos,
                PoseUtils.matrix_from_quat(object_quat),
            ),
            "destination_square": PoseUtils.make_pose(
                destination_pos,
                PoseUtils.matrix_from_quat(destination_quat),
            ),
        }

    # ------------------------------------------------------------------
    # 6) subtask signals (no-op defaults)
    # ------------------------------------------------------------------
    # Tasks that want automatic subtask annotation should override these in a
    # task-specific BaseILEnv subclass. The empty-dict defaults keep the
    # annotated Mimic recorder happy while leaving subtask annotation to be
    # done manually.
    def get_subtask_start_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        return {}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        return self.get_subtask_term_predicates(env_ids=env_ids)

    def get_subtask_term_predicates(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return raw 0/1 subtask predicates read from ``obs_buf["subtask_terms"]``.

        Names returned here must match the ``subtask_term_signal`` values declared
        in the task's Mimic cfg — the annotated-teleop recorder uses them as
        queue heads.
        """
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf.get("subtask_terms", {})
        return {name: tensor[env_ids] for name, tensor in subtask_terms.items()}
