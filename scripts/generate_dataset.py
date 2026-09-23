# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--use_skillgen",
    action="store_true",
    default=False,
    help="use skillgen to generate motion trajectories",
)
parser.add_argument(
    "--dataset_schema",
    choices=["legacy", "standard"],
    default="legacy",
    help="Dataset schema to use for the generated output file.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version
    # installed by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import inspect
import logging
import random
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import StandardGeneratedMimicRecorderManagerCfg
from isaaclab.managers import RecorderTermCfg
from isaaclab.utils.math import make_pose, matrix_from_quat

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

from isaaclab_mimic.datagen.generation import env_loop, run_data_generator, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401
import so101_chess.tasks  # noqa: F401
from so101_chess.tasks.manager_based.so101_chess.pickup_datagen import (
    ChessDataGenInfoPool,
    ChessPickupDataGenerator,
)
from so101_chess.tasks.manager_based.so101_chess.recorders import ActivePieceInitialStateRecorder

# import logger
logger = logging.getLogger(__name__)


class DestinationSquareInitialStateRecorder(ActivePieceInitialStateRecorder):
    """Record the virtual destination beside the active piece at reset."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        key, initial_state = super().record_post_reset(env_ids)
        env = self._env
        if env_ids is None:
            resolved_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        elif isinstance(env_ids, slice):
            resolved_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
        else:
            resolved_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)

        # Mimic exposes this virtual square in the robot-root frame. Standard
        # initial_state rigid-object poses use the environment-origin frame.
        destination_pose_b = env.get_object_poses(env_ids=resolved_env_ids)["destination_square"]
        robot = env.scene["robot"]
        robot_pose_env = make_pose(
            robot.data.root_pos_w[resolved_env_ids] - env.scene.env_origins[resolved_env_ids],
            matrix_from_quat(robot.data.root_quat_w[resolved_env_ids]),
        )
        destination_pose_env = robot_pose_env @ destination_pose_b
        initial_state["rigid_objects"]["destination_square"] = {
            "initial_pose": destination_pose_env,
            "scale": torch.ones(
                (len(resolved_env_ids), 3), dtype=destination_pose_env.dtype, device=destination_pose_env.device
            ),
        }
        return key, initial_state


def setup_chess_async_generation(
    env,
    num_envs: int,
    input_file: str,
    success_term,
    pause_subtask: bool = False,
    motion_planners=None,
    failure_terms=None,
):
    """Create the standard async workers with chess pickup metadata and latching."""
    event_loop = asyncio.get_event_loop()
    reset_queue = asyncio.Queue()
    action_queue = asyncio.Queue()
    info_pool_lock = asyncio.Lock()
    info_pool = ChessDataGenInfoPool(
        env,
        env.cfg,
        env.device,
        asyncio_lock=info_pool_lock,
    )
    info_pool.load_from_dataset_file(input_file)
    pickup_types = sorted(set(info_pool.piece_types))
    print(
        f"Loaded {info_pool.num_datagen_infos} to chess datagen info pool; "
        f"generation reset types: {pickup_types}"
    )

    data_generator = ChessPickupDataGenerator(env=env, src_demo_datagen_info_pool=info_pool)
    tasks = []
    for env_id in range(num_envs):
        motion_planner = motion_planners[env_id] if motion_planners else None
        tasks.append(
            event_loop.create_task(
                run_data_generator(
                    env,
                    env_id,
                    reset_queue,
                    action_queue,
                    data_generator,
                    success_term,
                    pause_subtask=pause_subtask,
                    motion_planner=motion_planner,
                    failure_terms=failure_terms,
                )
            )
        )

    return {
        "tasks": tasks,
        "event_loop": event_loop,
        "reset_queue": reset_queue,
        "action_queue": action_queue,
        "info_pool": info_pool,
    }


def main():
    num_envs = args_cli.num_envs

    # Setup output paths and get env name
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    # Configure environment
    recorder_cfg = StandardGeneratedMimicRecorderManagerCfg() if args_cli.dataset_schema == "standard" else None
    env_cfg, success_term, failure_terms = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
        recorder_cfg=recorder_cfg,
    )
    if recorder_cfg is not None and hasattr(env_cfg.events, "reset_sobol_chess_move"):
        # The chess environment normally installs its active-piece recorder.
        # A generic RecorderTermCfg keeps this local extension in place while
        # preserving the task's other recorder and HDF5 customizations.
        env_cfg.recorders.record_initial_state = RecorderTermCfg(class_type=DestinationSquareInitialStateRecorder)

    # Create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # Check if the mimic API from this environment contains decprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        logger.warning(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # Set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # Reset before starting
    env.reset()

    motion_planners = None
    if args_cli.use_skillgen:
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        # Create one motion planner per environment
        motion_planners = {}
        for env_id in range(num_envs):
            print(f"Initializing motion planner for environment {env_id}")
            # Create a config instance from the task name
            planner_config = CuroboPlannerCfg.from_task_name(env_name)

            # Ensure visualization is only enabled for the first environment
            # If not, sphere and plan visualization will be too slow in isaac lab
            # It is efficient to visualize the spheres and plan for the first environment in rerun
            if env_id != 0:
                planner_config.visualize_spheres = False
                planner_config.visualize_plan = False

            motion_planners[env_id] = CuroboPlanner(
                env=env,
                robot=env.scene["robot"],
                config=planner_config,  # Pass the config object
                env_id=env_id,  # Pass environment ID
            )

        env.cfg.datagen_config.use_skillgen = True

    # Setup and run async data generation
    async_components = setup_chess_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
        motion_planners=motion_planners,  # Pass the motion planners dictionary
        failure_terms=failure_terms,
    )

    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        # Cancel all async tasks when env_loop finishes
        data_gen_tasks.cancel()
        try:
            # Wait for tasks to be cancelled
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled and cleaned up.")
        except Exception as e:
            print(f"Error cancelling remaining async tasks: {e}")
        # Cleanup of motion planners and their visualizers
        if motion_planners is not None:
            for env_id, planner in motion_planners.items():
                if getattr(planner, "plan_visualizer", None) is not None:
                    print(f"Closing plan visualizer for environment {env_id}")
                    planner.plan_visualizer.close()
                    planner.plan_visualizer = None
            motion_planners.clear()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # Close sim app
    simulation_app.close()
