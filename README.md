# IL Task: So101Chess

Auto-generated IsaacLab IL extension for the **LeIsaac-SO101-Chess-v0** task.

## Installation

```bash
cd /home/roboticslab/IsaacTools/IsaacTasks/so101_chess
/home/roboticslab/IsaacTools/.venv/bin/python -m pip install --no-build-isolation --no-deps -e source/so101_chess
```

This task uses the IsaacTools environment with Isaac Lab, LeIsaac, and
`single_so101_cubes_leisaac` installed. USD assets are tracked with Git LFS;
after cloning this repository, run `git lfs pull` before launching the task.
Generated datasets stay local under `datasets/` and are not tracked by Git.

The editable install writes `so101_chess_register.pth` into your
site-packages, which auto-imports `so101_chess` on Python startup. That
import side-effects `gym.register(...)` for `LeIsaac-SO101-Chess-v0` and `LeIsaac-SO101-Chess-v0-Mimic`,
so IsaacLab's stock launch scripts can find your task without any per-task
patching.

## Usage

Use LeIsaac's teleoperation launcher with a physical SO-101 leader. The task
switches from its default Cartesian Mimic action to six ordered joint targets
(five arm joints and one gripper joint) before the environment is created.

```bash
# Leader connected directly to this computer
cd /home/roboticslab/IsaacTools/leisaac
/home/roboticslab/IsaacTools/.venv/bin/python \
  scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-SO101-Chess-v0 \
  --enable_pinocchio \
  --teleop_device so101leader \
  --port /dev/ttyACM0 \
  --num_envs 1 \
  --enable_cameras
```

To use a leader connected to another computer, start the publisher on that
computer:

```bash
python scripts/environments/teleoperation/so101_joint_state_server.py \
  --port /dev/ttyACM0 \
  --id leader_arm \
  --rate 50
```

Then start the simulation with the publisher's IP address:

```bash
cd /home/roboticslab/IsaacTools/leisaac
/home/roboticslab/IsaacTools/.venv/bin/python \
  scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-SO101-Chess-v0 \
  --enable_pinocchio \
  --teleop_device so101leader \
  --remote_endpoint tcp://LEADER_COMPUTER_IP:5556 \
  --num_envs 1 \
  --enable_cameras
```

Add `--recalibrate` for a locally attached leader when its cached calibration
must be replaced. Use `--calibration_file /path/to/so101_leader.json` to load a
specific local calibration.

Other task utilities use the default Pink IK Cartesian action configuration.
Pinocchio must be enabled before Isaac Sim starts:

```bash
# Record physical-leader demonstrations by adding --record to either command above
# and optionally: --dataset_file ./datasets/so101_chess.hdf5

# Record annotated demos for the Mimic data-generation pipeline
/home/roboticslab/IsaacTools/.venv/bin/python scripts/record_annotated_demos.py \
  --task LeIsaac-SO101-Chess-v0-Mimic --enable_pinocchio --enable_cameras

```

The destination-side Mimic subtasks record both `active_piece` and the
virtual `destination_square` pose. Source datasets recorded before those two
object-pose streams were added must be recorded again before synthetic-data
generation; they do not contain enough information for multi-object neighbor
selection.

The ordered sequence includes `lift_after_place` between `place_object` and
`return_home`. It requires the open gripper to retreat 5 cm upward with no
more than 2 cm of horizontal drift while the piece remains on its destination
square. Source demonstrations must include this signal before they can drive
the seven-stage Mimic configuration.

The task also customizes the standard dataset recorder for RebelHDF5:

- `initial_state/rigid_objects` contains only the manipulated object, under the
  stable name `active_piece`; parked inactive chess pieces are not serialized.
- generated episodes retain Isaac Lab's canonical
  `reference_demo_indices/chess` values and also write the single-arm
  `reference_demo_indices/left` alias expected by RebelHDF5. Each array entry
  is the source `demo_<index>` selected for the corresponding subtask slot.

Files recorded before this change can be repaired without re-running a
simulation. For large camera datasets, in-place repair avoids copying the
image arrays:

```bash
/home/roboticslab/IsaacTools/.venv/bin/python \
  scripts/repair_rebelhdf5_dataset.py \
  datasets/annotated_dataset_2.hdf5 --in-place

/home/roboticslab/IsaacTools/.venv/bin/python \
  scripts/repair_rebelhdf5_dataset.py \
  datasets/chess_generated_standard_5.hdf5 --in-place

/home/roboticslab/IsaacTools/.venv/bin/python \
  scripts/repair_rebelhdf5_dataset.py \
  datasets/chess_generated_standard_5_failed.hdf5 --in-place
```

Use `--output repaired.hdf5` instead of `--in-place` to keep the original.

Use a launcher that exposes `--enable_pinocchio`; the stock random/zero-agent
scripts in this Isaac Lab checkout do not provide that early-import option.

The physical leader command is intentionally `--teleop_device so101leader` for
both local serial and remote ZMQ operation. The `--remote_endpoint` option is
what selects the network receiver.

## Evaluate the fine-tuned XVLA policy

The checkpoint at `checkpoints/last/pretrained_model` is the deployable model
directory. Run the two processes below in separate terminals. The inference
server uses the LeRobot Python environment; Isaac Sim uses the IsaacTools Python
environment. Neither LeRobot nor LeIsaac source needs to be edited.

```bash
cd /home/roboticslab/IsaacTools/IsaacTasks/so101_chess
/home/roboticslab/lerobot/.venv/bin/python scripts/xvla_inference_server.py
```

The checkpoint's `.safetensors` files must be readable by the account running
the server. If a newly exported checkpoint is owner-only, its owner can grant
the shared `robotics` group read access with
`chmod g+r /path/to/pretrained_model/*.safetensors`.

```bash
cd /home/roboticslab/IsaacTools/IsaacTasks/so101_chess
/home/roboticslab/IsaacTools/.venv/bin/python scripts/eval_xvla.py \
  --enable_cameras --episodes 10
```

Add `--headless` to run without a viewer. The task's control step is 1/60 s;
the runner executes all 30 actions from each 30-action prediction and then
requests a fresh prediction. Change `--action_horizon` to control how often it
replans. For a brief connection and camera check, pass `--episodes 1
--episode_length_s 5` to the Isaac command.

Evaluation uses fresh random chess moves on every reset, matching the Mimic
generator used for training. Use `--seed 42` for a repeatable sequence and
`--episode_length_s 10` for quicker policy checks. To inspect whether the
policy or the task controller is limiting motion, pass `--debug_steps 30`.
The runner will print policy targets, processed targets, and measured joints;
`--debug_observation /tmp/xvla-observation.npz` also saves the first live
camera pair, state, and predicted action chunk.

This bridge sends the top and wrist RGB cameras, six joint positions in
radians, and the exact dataset task text over localhost. It loads the
checkpoint's saved preprocessing and action unnormalization. The checkpoint
retains an eight-value state declaration and a third image slot from XVLA's
base config; its training dataset and normalization data have six joint values
and two cameras. XVLA pads the state internally and masks the missing camera.
The bridge passes camera frames at their native 640×480 resolution so XVLA's
own aspect-preserving resize matches training.
The runner also disables the physical leader's wrist calibration offset,
because the dataset actions are already final joint targets in radians.
It renders and refreshes both cameras 60 times after each reset before asking
the policy for an action. Isaac's RTX camera buffer can initially show the
previous chess move after an automatic reset. Use `--reset_camera_renders` to
change the number of extra renders.

To check the simulator independently of XVLA, replay recorded successful demo
0 with its original pawn asset:

```bash
/home/roboticslab/IsaacTools/.venv/bin/python scripts/validate_xvla_bridge.py \
  --headless --enable_cameras --episode 0 --piece pawn_white
```

This uses the same absolute-radian joint action configuration as evaluation
and reports task success, joint tracking error, and top/wrist camera error
against the HDF5 recording. The source file stores the piece under the generic
`active_piece` alias, so other episodes require identifying their original
piece asset before replay.

LeIsaac's stock `policy_inference.py --policy_type lerobot-xvla` path is not
compatible with this checkpoint as installed: it converts actions through
SO-101 motor units, and the installed LeRobot async server does not list XVLA
among supported policies. The local bridge calls XVLA and its saved processors
directly instead.

## Domain randomization

The chess task randomizes the scene at every reset:

- The three scene lights share a sampled intensity multiplier from 0.4 to 2.0 and a color temperature from 2500 K to 7000 K. The imported ground plane's much brighter sphere light is disabled so these changes affect camera images. Sampled values are printed as `[CHESS DR]` and stored in `env.unwrapped.chess_randomization_state`.
- The table stays in white shades (0.88–0.97) with a subtle painted MDF-like grain made from one Isaac Sim library texture, `textured_wall.png`. Only its grain contrast and roughness vary slightly.
- The pieces stay white or black and use Isaac Sim's `OmniSurface_Plastic` material family. Their shade and matte plastic roughness vary narrowly to model 3D-printed pieces.
- The robot uses a matte plastic finish based on the supplied blue-black filament hue `#080A0D`. Its shader reflectance is lifted slightly to retain visible detail under dim randomized lights, with a narrow brightness variation.
- Chessboard squares vary only between shades of white and dark gray. Their texture and the yellow border material are left alone. Source and destination markers remain red and green.
- Each episode contains 1–6 pieces by default. Other pieces are placed on separate squares away from the source, destination and move path; unused rigid bodies are parked off the board.

Change the light ranges in `So101ChessEventCfg.randomize_appearance.params` and the piece-count range in `So101ChessEventCfg.reset_sobol_chess_move.params`. Setting both piece-count values to `1` restores single-piece episodes. The current initial-state recorder exports only the active piece, so its HDF5 state does not reconstruct distractor placement or visual randomization.

## Project Structure

```
source/so101_chess/so101_chess/
├── assets/          # USD scene, object, and robot files
├── base_il_env/     # Base IL environment (shared across tasks)
└── tasks/           # Task-specific configurations and MDP functions
```

## Customization

- Edit `tasks/manager_based/so101_chess/so101_chess_cfg.py` for task configuration
- Add custom MDP functions in `tasks/manager_based/so101_chess/mdp/`
- Place USD files in the `assets/` subdirectories
