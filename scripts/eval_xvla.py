"""Run the chess XVLA policy in Isaac Sim using the local inference server."""

from __future__ import annotations

import argparse
import io
import multiprocessing
import urllib.request
from pathlib import Path

import numpy as np

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

# This task imports Pink, so Pinocchio must load before Isaac Sim libraries.
import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="LeIsaac-SO101-Chess-v0")
parser.add_argument("--policy_url", default="http://127.0.0.1:8765/infer")
parser.add_argument("--instruction", default="Move the chess piece from the red square to the green square")
parser.add_argument("--action_horizon", type=int, default=30, help="Steps to execute before requesting a new chunk.")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--reset_camera_renders", type=int, default=60, help="Extra renders to flush camera history after each reset.")
parser.add_argument("--request_timeout", type=float, default=120.0)
parser.add_argument("--debug_steps", type=int, default=0, help="Print policy, applied target, and measured joints for N steps.")
parser.add_argument("--debug_observation", type=Path, default=None, help="Save the first live observation and action chunk as NPZ.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("camera observations require --enable_cameras")
if not 1 <= args.action_horizon <= 30:
    parser.error("--action_horizon must be between 1 and 30")
if args.episodes < 1:
    parser.error("--episodes must be at least 1")
if args.reset_camera_renders < 0:
    parser.error("--reset_camera_renders must be nonnegative")

simulation_app = AppLauncher(vars(args)).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import so101_chess.tasks  # noqa: F401


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAMERA_NAMES = ("top_camera", "right_wrist_camera")
SUBTASK_STAGES = (
    "move_over_source",
    "grasp_object",
    "lift_object",
    "move_over_destination",
    "place_object",
    "lift_after_place",
    "return_home",
)


def refresh_reset_cameras(env) -> None:
    """Drain RTX's old camera frame and material history after a chess reset."""
    for _ in range(args.reset_camera_renders):
        env.sim.render()
        for name in CAMERA_NAMES:
            env.scene[name].update(0.0, force_recompute=True)


def infer(env, joint_ids: list[int], save_observation: Path | None = None) -> np.ndarray:
    state = env.scene["robot"].data.joint_pos[0, joint_ids].detach().cpu().numpy().astype(np.float32)
    images = {}
    for name in CAMERA_NAMES:
        image = env.scene[name].data.output["rgb"][0, :, :, :3]
        pixels = image.detach().cpu().numpy()
        if np.issubdtype(pixels.dtype, np.floating):
            if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
                raise ValueError(f"{name} RGB floats must be finite values in [0, 1]")
            pixels = np.rint(pixels * 255)
        images[name] = pixels.astype(np.uint8)
    payload = io.BytesIO()
    np.savez(payload, state=state, task=args.instruction, **images)
    request = urllib.request.Request(
        args.policy_url,
        payload.getvalue(),
        {"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        actions = np.load(io.BytesIO(response.read()), allow_pickle=False)
    if actions.ndim != 2 or actions.shape[1] != 6 or not np.isfinite(actions).all():
        raise ValueError(f"Invalid policy actions: {actions.shape}")
    if save_observation is not None:
        save_observation.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_observation, state=state, actions=actions, task=args.instruction, **images)
    return actions


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.use_teleop_device("so101leader")
    # The training data stores final joint targets in radians. The leader-only
    # wrist calibration must not be added a second time during policy rollout.
    env_cfg.actions.arm_action.offset = 0.0
    env_cfg.recorders = None
    # The generated training demonstrations used random moves and did not
    # repeat a failed move. The base teleop task uses success-gated Sobol draws.
    env_cfg.events.reset_sobol_chess_move.params["sampling_strategy"] = "random"
    env_cfg.events.reset_sobol_chess_move.params["advance_on_success_only"] = False
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s
    if args.seed is not None:
        env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    try:
        robot = env.scene["robot"]
        joint_ids, found_names = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(found_names) != JOINT_NAMES or env.action_manager.total_action_dim != 6:
            raise ValueError(f"Unexpected robot/action layout: {found_names}")
        joint_limits = robot.data.soft_joint_pos_limits[0, joint_ids].detach().cpu().numpy()
        env.reset()
        refresh_reset_cameras(env)
        print("Isaac chess environment ready; requesting XVLA actions.", flush=True)
        completed = successes = 0
        step_count = 0
        episode_steps = 0
        episode_start = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().copy()
        max_arm_excursion = 0.0
        max_stage = 0
        while simulation_app.is_running() and completed < args.episodes:
            save_observation = args.debug_observation if step_count == 0 else None
            chunk = infer(env, joint_ids, save_observation)
            if step_count < args.debug_steps:
                state = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
                print(
                    f"chunk at step {step_count}: state={np.round(state, 3)} "
                    f"first={np.round(chunk[0], 3)} last={np.round(chunk[-1], 3)}",
                    flush=True,
                )
            for row in chunk[: args.action_horizon]:
                target = np.clip(row, joint_limits[:, 0], joint_limits[:, 1])
                action = torch.as_tensor(target, dtype=torch.float32, device=env.device).unsqueeze(0)
                with torch.inference_mode():
                    _, _, terminated, timed_out, _ = env.step(action)
                actual = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
                if not (terminated[0] or timed_out[0]):
                    max_arm_excursion = max(
                        max_arm_excursion, float(np.linalg.norm(actual[:5] - episode_start[:5]))
                    )
                    stage = getattr(env, "_chess_subtask_stage", None)
                    if stage is not None:
                        max_stage = max(max_stage, int(stage[0]))
                if step_count < args.debug_steps:
                    processed = np.concatenate(
                        [
                            env.action_manager.get_term(term).processed_actions[0].detach().cpu().numpy()
                            for term in ("arm_action", "gripper_action")
                        ]
                    )
                    print(
                        f"step {step_count}: policy={np.round(row, 3)} "
                        f"clipped={np.round(target, 3)} processed={np.round(processed, 3)} "
                        f"actual={np.round(actual, 3)}",
                        flush=True,
                    )
                step_count += 1
                episode_steps += 1
                if terminated[0] or timed_out[0]:
                    success = bool(env.termination_manager.get_term("success")[0])
                    failed = bool(env.termination_manager.get_term("failed")[0])
                    result = "success" if success else "failed" if failed else "timed out"
                    successes += int(success)
                    completed += 1
                    print(
                        f"Episode {completed}: {result} after {episode_steps} steps; "
                        f"max arm excursion {max_arm_excursion:.2f} rad; "
                        f"next subtask {SUBTASK_STAGES[max_stage] if max_stage < len(SUBTASK_STAGES) else 'all complete'}; "
                        f"{successes} successes",
                        flush=True,
                    )
                    episode_start = actual.copy()
                    max_arm_excursion = 0.0
                    max_stage = 0
                    episode_steps = 0
                    if completed < args.episodes:
                        refresh_reset_cameras(env)
                    break
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
