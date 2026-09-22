"""Replay one successful HDF5 demonstration in the current chess environment.

This checks the same six-joint action path used by eval_xvla.py. The source
dataset stores the active chess piece under an alias, so specify its actual
asset with --piece. For demo_0, the piece is pawn_white.
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="datasets/chess_generated_100ep_secondtry.hdf5")
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--piece", required=True, help="Scene asset for the active piece, e.g. pawn_white for demo_0.")
parser.add_argument("--save_first_image", type=Path, default=None, help="Write the first replayed top RGB frame as a PNG.")
parser.add_argument("--save_reset_images", type=Path, default=None, help="Write camera frames at several render counts after auto reset.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("camera comparison requires --enable_cameras")
simulation_app = AppLauncher(vars(args)).app

import gymnasium as gym
import h5py
import numpy as np
import torch
from PIL import Image
from isaaclab.utils.math import quat_from_matrix
from isaaclab_tasks.utils import parse_env_cfg

import so101_chess.tasks  # noqa: F401
from so101_chess.tasks.manager_based.so101_chess.mdp.board import square_surface_position


PIECES = ("pawn_white", "rook_white", "knight_white", "bishop_white", "queen_white", "king_white")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def nearest_square(env, position):
    squares = [
        ((r, c), np.asarray(square_surface_position(env.sim.stage, 0, r, c), dtype=np.float64))
        for r in range(8) for c in range(8)
    ]
    square, center = min(squares, key=lambda item: np.linalg.norm(item[1][:2] - position[:2]))
    error = float(np.linalg.norm(center[:2] - position[:2]))
    if error > 0.01:
        raise ValueError(f"Recorded position {position} is {error:.3f} m from nearest square {square}")
    return square


def reset_to_demo(env, name, source, target, joint_position, joint_velocity, piece_matrix):
    print(f"Resetting with piece {name}", flush=True)
    env._chess_move_reset_state = {"count": 0, "move": (name, source, target)}
    env.reset()
    print(f"Reset completed for {name}", flush=True)
    robot = env.scene["robot"]
    position = torch.as_tensor(joint_position[None], device=env.device)
    velocity = torch.as_tensor(joint_velocity[None], device=env.device)
    robot.write_joint_state_to_sim(position, velocity)
    robot.set_joint_position_target(position)
    piece = env.scene[name]
    matrix = torch.as_tensor(piece_matrix[None], device=env.device)
    quaternion = quat_from_matrix(matrix[:, :3, :3])
    pose = torch.cat((matrix[:, :3, 3], quaternion), dim=-1)
    piece.write_root_pose_to_sim(pose)
    piece.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    env.sim.forward()


def image(env, name="top_camera"):
    return env.scene[name].data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)


def main():
    with h5py.File(args.dataset) as file:
        demo = file[f"data/demo_{args.episode}"]
        if not bool(demo.attrs["success"]):
            raise ValueError("Select a recorded successful episode")
        actions = demo["actions/joints"][:]
        states = demo["obs/articulations/robot/joint_position"][:]
        recorded_first = demo["obs/cameras/top_camera"][0]
        recorded_first_wrist = demo["obs/cameras/right_wrist_camera"][0]
        joint_position = demo["initial_state/articulations/robot/joint_position"][0]
        joint_velocity = demo["initial_state/articulations/robot/joint_velocity"][0]
        piece_matrix = demo["initial_state/rigid_objects/active_piece/initial_pose"][0]
        target_matrix = demo["initial_state/rigid_objects/destination_square/initial_pose"][0]

    cfg = parse_env_cfg("LeIsaac-SO101-Chess-v0", device=args.device, num_envs=1)
    cfg.use_teleop_device("so101leader")
    cfg.actions.arm_action.offset = 0.0
    cfg.recorders = None
    cfg.events.reset_sobol_chess_move.params["advance_on_success_only"] = True
    cfg.episode_length_s = max(cfg.episode_length_s, (len(actions) + 30) / 60)
    env = gym.make("LeIsaac-SO101-Chess-v0", cfg=cfg).unwrapped
    try:
        source = nearest_square(env, piece_matrix[:3, 3])
        target = nearest_square(env, target_matrix[:3, 3])
        print(f"Recorded demo_{args.episode}: {len(actions)} steps, source={source}, target={target}", flush=True)
        robot = env.scene["robot"]
        joint_ids, names = robot.find_joints(list(JOINTS), preserve_order=True)
        if tuple(names) != JOINTS:
            raise ValueError(f"Unexpected joint order: {names}")

        if args.piece not in PIECES:
            raise ValueError(f"--piece must be one of {PIECES}")
        reset_to_demo(env, args.piece, source, target, joint_position, joint_velocity, piece_matrix)
        print(f"Replaying as {args.piece}", flush=True)

        errors = []
        success = failure = timeout = False
        first_image_error = None
        first_wrist_error = None
        for step, values in enumerate(actions):
            action = torch.as_tensor(values[None], device=env.device)
            with torch.inference_mode():
                _, _, terminated, timed_out, _ = env.step(action)
            if terminated[0] or timed_out[0]:
                # Isaac Lab has already reset the environment at this point.
                success = bool(env.termination_manager.get_term("success")[0])
                failure = bool(env.termination_manager.get_term("failed")[0])
                timeout = bool(timed_out[0])
                if args.save_reset_images is not None:
                    args.save_reset_images.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image(env)).save(args.save_reset_images.with_name(args.save_reset_images.stem + "-0.png"))
                    for render_count in range(1, 61):
                        env.sim.render()
                        env.scene["top_camera"].update(0.0, force_recompute=True)
                        if render_count in (1, 2, 3, 5, 10, 20, 30, 40, 60):
                            Image.fromarray(image(env)).save(
                                args.save_reset_images.with_name(args.save_reset_images.stem + f"-{render_count}.png")
                            )
                print(f"step={step} episode ended", flush=True)
                break
            actual = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
            errors.append(np.abs(actual - states[step]))
            if step == 0:
                first_image = image(env)
                first_image_error = float(np.abs(first_image.astype(np.int16) - recorded_first.astype(np.int16)).mean())
                first_wrist_error = float(
                    np.abs(image(env, "right_wrist_camera").astype(np.int16) - recorded_first_wrist.astype(np.int16)).mean()
                )
                if args.save_first_image is not None:
                    args.save_first_image.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(first_image).save(args.save_first_image)
            if step % 60 == 0:
                print(f"step={step} joint_mae={np.mean(errors[-1]):.4f} rad", flush=True)
        errors = np.asarray(errors)
        max_step, max_joint = np.unravel_index(np.argmax(errors), errors.shape)
        print(
            f"Replay result: success={success} failed={failure} timeout={timeout}; "
            f"steps={step + 1}/{len(actions)}; mean joint MAE={errors.mean():.4f} rad; "
            f"max joint error={errors.max():.4f} rad ({JOINTS[max_joint]} at step {max_step}); "
            f"first top/wrist camera MAE={first_image_error:.2f}/{first_wrist_error:.2f} out of 255",
            flush=True,
        )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
