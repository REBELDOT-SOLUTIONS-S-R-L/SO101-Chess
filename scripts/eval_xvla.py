"""Run the chess XVLA policy in Isaac Sim using the local inference server."""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import os
import urllib.request
from datetime import datetime, timezone
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
parser.add_argument(
    "--log_file",
    type=Path,
    default=None,
    help="Append run and per-episode records as JSON Lines, syncing each finished episode to disk.",
)
parser.add_argument(
    "--checkpoint_label",
    default=None,
    help="Checkpoint identifier stored in --log_file records (the policy is still served by --policy_url).",
)
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
from so101_chess.tasks.manager_based.so101_chess import mdp


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAMERA_NAMES = ("top_camera", "right_wrist_camera")


class JsonlRunLogger:
    """Append durable, machine-readable evaluation records."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")

    def write(self, record: dict, *, sync: bool = False) -> None:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        self._stream.write(json.dumps(payload, sort_keys=True) + "\n")
        self._stream.flush()
        if sync:
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        self._stream.close()


def current_chess_move(env) -> dict[str, object]:
    """Return the reset move currently assigned to the single evaluation env."""
    state = getattr(env, "_chess_move_reset_state", None)
    move = state.get("move") if isinstance(state, dict) else None
    if not isinstance(move, (tuple, list)) or len(move) != 3:
        return {}
    piece_name, source_square, target_square = move
    return {
        "piece_name": str(piece_name),
        "piece_type": str(piece_name).split("_", maxsplit=1)[0],
        "source_square": [int(value) for value in source_square],
        "target_square": [int(value) for value in target_square],
    }


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
    run_logger = JsonlRunLogger(args.log_file) if args.log_file is not None else None
    env = None
    completed = successes = 0
    try:
        if run_logger is not None:
            run_logger.write(
                {
                    "event": "run_start",
                    "checkpoint": args.checkpoint_label,
                    "task": args.task,
                    "instruction": args.instruction,
                    "policy_url": args.policy_url,
                    "requested_episodes": args.episodes,
                    "episode_length_s": args.episode_length_s,
                    "action_horizon": args.action_horizon,
                    "seed": args.seed,
                },
                sync=True,
            )
            print(f"Writing episode records to {run_logger.path}", flush=True)

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.use_teleop_device("so101leader")
        # The training data stores final joint targets in radians. The leader-only
        # wrist calibration must not be added a second time during policy rollout.
        env_cfg.actions.arm_action.offset = 0.0
        env_cfg.recorders = None
        # Policy evaluation is outcome-only. Do not compute the Mimic subtask
        # predicates, do not terminate on their failure latch, and do not require
        # the sequential subtask stage (or a closed gripper) for final success.
        env_cfg.observations.subtask_terms = None
        env_cfg.terminations.failed = None
        env_cfg.terminations.success.func = mdp.chess_move_final_success
        env_cfg.terminations.success.params = {
            "target_xy_threshold": 0.02,
            "target_z_threshold": 0.02,
            "max_tilt_deg": 5.0,
            "home_eef_link": "gripper_frame_link",
            "home_pos": (0.05, -0.12, 0.0135),
            "home_threshold": (0.04, 0.04, 0.03),
            "min_hold_steps": 5,
        }
        # The generated training demonstrations used random moves and did not
        # repeat a failed move. The base teleop task uses success-gated Sobol draws.
        env_cfg.events.reset_sobol_chess_move.params["sampling_strategy"] = "random"
        env_cfg.events.reset_sobol_chess_move.params["advance_on_success_only"] = False
        if args.episode_length_s is not None:
            env_cfg.episode_length_s = args.episode_length_s
        if args.seed is not None:
            env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        robot = env.scene["robot"]
        joint_ids, found_names = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(found_names) != JOINT_NAMES or env.action_manager.total_action_dim != 6:
            raise ValueError(f"Unexpected robot/action layout: {found_names}")
        joint_limits = robot.data.soft_joint_pos_limits[0, joint_ids].detach().cpu().numpy()
        env.reset()
        refresh_reset_cameras(env)
        print("Isaac chess environment ready; requesting XVLA actions.", flush=True)
        step_count = 0
        episode_steps = 0
        episode_move = current_chess_move(env)
        episode_start = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().copy()
        max_arm_excursion = 0.0
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
                    result = "success" if success else "timed out"
                    successes += int(success)
                    completed += 1
                    print(
                        f"Episode {completed}: {result} after {episode_steps} steps; "
                        f"max arm excursion {max_arm_excursion:.2f} rad; "
                        f"{successes} successes",
                        flush=True,
                    )
                    if run_logger is not None:
                        run_logger.write(
                            {
                                "event": "episode",
                                "checkpoint": args.checkpoint_label,
                                "episode": completed,
                                "success": success,
                                "result": result,
                                "terminated": bool(terminated[0]),
                                "timed_out": bool(timed_out[0]),
                                "episode_steps": episode_steps,
                                "simulated_duration_s": episode_steps * float(env.step_dt),
                                "global_steps": step_count,
                                "max_arm_excursion_rad": max_arm_excursion,
                                "cumulative_successes": successes,
                                "cumulative_success_rate": successes / completed,
                                **episode_move,
                            },
                            sync=True,
                        )
                    episode_start = actual.copy()
                    max_arm_excursion = 0.0
                    episode_steps = 0
                    if completed < args.episodes:
                        refresh_reset_cameras(env)
                        episode_move = current_chess_move(env)
                    break
        if completed != args.episodes:
            raise RuntimeError(
                f"Evaluation stopped after {completed}/{args.episodes} episodes before completing the run"
            )
        if run_logger is not None:
            run_logger.write(
                {
                    "event": "run_complete",
                    "checkpoint": args.checkpoint_label,
                    "completed_episodes": completed,
                    "successes": successes,
                    "success_rate": successes / completed if completed else 0.0,
                },
                sync=True,
            )
    except BaseException as exc:
        if run_logger is not None:
            run_logger.write(
                {
                    "event": "run_error",
                    "checkpoint": args.checkpoint_label,
                    "completed_episodes": completed,
                    "successes": successes,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sync=True,
            )
        raise
    finally:
        if env is not None:
            env.close()
        if run_logger is not None:
            run_logger.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
