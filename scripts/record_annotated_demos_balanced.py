# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Record type-balanced SO-101 chess Mimic demonstrations with online annotations.

The script records a standard-schema annotated Mimic dataset directly during
teleoperation.  Each end-effector owns an independent queue derived from
``env.cfg.subtask_configs``.  At every step only the head signal of each queue
is checked against raw environment predicates.  When a head signal remains true
for its configured dwell, that signal is latched and only that queue advances.

Unlike the standard recorder, this variant cycles uniformly through the
requested normalized chess-piece types. Each type owns an independently
indexed Sobol stream over its legal source/destination moves. With the default
six-type configuration, every block of six successfully saved episodes
contains one episode of each type. ``--piece_type`` restricts collection to a
single type while retaining that type's original Sobol sequence.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import logging
import math
import os
import time
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record standard Mimic demos with online annotations.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here or via the environment config. Built-ins: keyboard, spacemouse, gamepad; "
        "LeIsaac: so101leader."
    ),
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/annotated_dataset_balanced.hdf5",
    help="File path to export recorded annotated demos.",
)
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping/recording rate in Hz.")
parser.add_argument(
    "--port",
    type=str,
    default="/dev/ttyACM0",
    help="Serial port for LeIsaac's SO-101 leader.",
)
parser.add_argument(
    "--calibration_file",
    type=str,
    default=None,
    help="Optional SO-101 leader calibration JSON; otherwise LeIsaac's default cache is used.",
)
parser.add_argument(
    "--recalibrate",
    action="store_true",
    help="Recalibrate the SO-101 leader before recording.",
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=60,
    help="Number of demonstrations to record. The default records 10 successful episodes per piece type.",
)
parser.add_argument(
    "--piece_type",
    choices=("pawn", "rook", "knight", "bishop", "queen", "king"),
    default=None,
    help="Record only one normalized piece type instead of cycling through all six.",
)
parser.add_argument(
    "--sobol_start_index",
    type=int,
    default=0,
    help=(
        "Per-piece Sobol draw index at which normal balanced recording starts. "
        "Use 10 to extend an existing dataset that already contains ten draws per type."
    ),
)
parser.add_argument(
    "--sobol_plan",
    type=str,
    default=None,
    metavar="TYPE:INDEX,...",
    help=(
        "Record an exact ordered plan of per-type Sobol draws, for example "
        "'rook:0,knight:2,knight:5'. This is mutually exclusive with "
        "--piece_type and --sobol_start_index."
    ),
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Teleop sensitivity factor.")
parser.add_argument(
    "--default_signal_dwell",
    type=int,
    default=3,
    help="Default consecutive true steps required before latching a subtask signal.",
)
parser.add_argument(
    "--signal_dwell",
    action="append",
    default=[],
    metavar="SIGNAL=STEPS",
    help="Per-signal dwell override. Can be passed multiple times.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.sobol_start_index < 0:
    parser.error("--sobol_start_index must be non-negative")
if args_cli.sobol_plan and args_cli.piece_type is not None:
    parser.error("--sobol_plan cannot be combined with --piece_type")
if args_cli.sobol_plan and args_cli.sobol_start_index != 0:
    parser.error("--sobol_plan cannot be combined with a nonzero --sobol_start_index")

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

if "handtracking" in args_cli.teleop_device.lower():
    args_cli.xr = True
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import StandardAnnotatedMimicRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.utils.math import make_pose, matrix_from_quat

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from so101_chess.tasks.manager_based.so101_chess.mdp import events as chess_events
from so101_chess.tasks.manager_based.so101_chess.piece_types import normalize_piece_type

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BALANCED_PIECE_TYPES = ("pawn", "rook", "knight", "bishop", "queen", "king")


def parse_sobol_plan(value: str | None) -> tuple[tuple[str, int], ...]:
    """Parse an ordered ``piece_type:draw_index`` recording plan."""
    if value is None:
        return ()
    entries: list[tuple[str, int]] = []
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        piece_type, separator, raw_index = entry.partition(":")
        piece_type = piece_type.strip().lower()
        if separator != ":" or piece_type not in BALANCED_PIECE_TYPES:
            raise ValueError(
                f"Invalid Sobol plan entry {entry!r}; expected TYPE:INDEX with TYPE in "
                f"{BALANCED_PIECE_TYPES}"
            )
        try:
            draw_index = int(raw_index)
        except ValueError as exc:
            raise ValueError(f"Invalid Sobol draw index in {entry!r}") from exc
        if draw_index < 0:
            raise ValueError(f"Sobol draw index must be non-negative in {entry!r}")
        entries.append((piece_type, draw_index))
    if not entries:
        raise ValueError("--sobol_plan did not contain any entries")
    return tuple(entries)


try:
    RECORDING_MOVE_PLAN = parse_sobol_plan(args_cli.sobol_plan)
except ValueError as exc:
    parser.error(str(exc))

RECORDING_PIECE_TYPES = (
    tuple(piece_type for piece_type, _ in RECORDING_MOVE_PLAN)
    if RECORDING_MOVE_PLAN
    else ((args_cli.piece_type,) if args_cli.piece_type is not None else BALANCED_PIECE_TYPES)
)
_ACTIVE_SOBOL_DRAW_INDEX: int | None = None


def sample_indexed_piece_sobol_move(
    seed: int | None = None,
    piece_names: list[str] | tuple[str, ...] | None = None,
    board_size: int = 8,
    available_piece_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Draw piece color, source square and legal target from a 3-D Sobol stream."""
    if seed is None:
        raise ValueError("Balanced recording requires a deterministic Sobol index")
    candidate_names = list(piece_names or chess_events._DEFAULT_CHESS_PIECES)
    if available_piece_names is not None:
        candidate_names = [name for name in candidate_names if name in available_piece_names]
    if not candidate_names:
        raise ValueError("Balanced Sobol sampling received no available piece names")

    selected_types = {normalize_piece_type(name) for name in candidate_names}
    if len(selected_types) != 1:
        raise ValueError(
            "Balanced Sobol sampling expects exactly one normalized piece type, "
            f"got {sorted(selected_types)}"
        )

    piece_type = next(iter(selected_types))
    type_index = BALANCED_PIECE_TYPES.index(piece_type)
    # In planned recording the reset wrapper selects an exact per-type draw.
    # Otherwise each type appears once per balanced cycle; in a single-type
    # recording every global sequence index belongs to that type.
    if _ACTIVE_SOBOL_DRAW_INDEX is not None:
        draw_index = _ACTIVE_SOBOL_DRAW_INDEX
    else:
        draw_index = int(seed) // len(RECORDING_PIECE_TYPES)
    engine = torch.quasirandom.SobolEngine(3, scramble=True, seed=type_index)
    if draw_index > 0:
        engine.fast_forward(draw_index)
    sample = engine.draw(1)[0]

    piece_name = candidate_names[int(float(sample[0].item()) * len(candidate_names))]
    source_squares = []
    targets_by_source = {}
    for row in range(board_size):
        for column in range(board_size):
            source = (row, column)
            targets = chess_events.get_valid_chess_targets(piece_name, source, board_size)
            if targets:
                source_squares.append(source)
                targets_by_source[source] = targets
    if not source_squares:
        raise ValueError(f"No legal source squares are available for {piece_name!r}")

    source_square = source_squares[int(float(sample[1].item()) * len(source_squares))]
    targets = targets_by_source[source_square]
    target_square = targets[int(float(sample[2].item()) * len(targets))]
    return piece_name, source_square, target_square


def sample_type_balanced_chess_move_reset(
    env: ManagerBasedRLEnv,
    env_ids,
    piece_names: list[str] | tuple[str, ...] | None = None,
    seed: int = 0,
    board_size: int = 8,
    z_offset: float = 0.001,
    advance_on_success_only: bool = True,
    sampling_strategy: str = "sobol",
    min_pieces: int = 1,
    max_pieces: int = 6,
):
    """Select piece type uniformly, then delegate reset and retry handling."""
    if sampling_strategy != "sobol":
        raise ValueError("The balanced recorder only supports Sobol move sampling")

    configured_names = list(piece_names or chess_events._DEFAULT_CHESS_PIECES)
    scene_keys = set(env.scene.keys())
    available_names = [name for name in configured_names if name in scene_keys]
    missing_types = [
        piece_type
        for piece_type in dict.fromkeys(RECORDING_PIECE_TYPES)
        if not any(normalize_piece_type(name) == piece_type for name in available_names)
    ]
    if missing_types:
        raise ValueError(
            "Recording requires every requested normalized piece type; "
            f"missing {missing_types}"
        )

    state = getattr(env, "_chess_move_reset_state", None)
    successful_move_count = int(state.get("count", 0)) if state is not None else 0
    global _ACTIVE_SOBOL_DRAW_INDEX
    if RECORDING_MOVE_PLAN:
        if successful_move_count >= len(RECORDING_MOVE_PLAN):
            raise RuntimeError(
                "The exact Sobol recording plan is exhausted; stop recording after "
                f"{len(RECORDING_MOVE_PLAN)} successful demonstrations"
            )
        selected_type, _ACTIVE_SOBOL_DRAW_INDEX = RECORDING_MOVE_PLAN[successful_move_count]
    else:
        _ACTIVE_SOBOL_DRAW_INDEX = None
        sequence_index = int(seed) + successful_move_count
        selected_type = RECORDING_PIECE_TYPES[sequence_index % len(RECORDING_PIECE_TYPES)]
    selected_names = [
        name for name in available_names if normalize_piece_type(name) == selected_type
    ]

    return chess_events.sample_chess_move_reset(
        env=env,
        env_ids=env_ids,
        piece_names=selected_names,
        seed=seed,
        board_size=board_size,
        z_offset=z_offset,
        advance_on_success_only=advance_on_success_only,
        sampling_strategy="sobol",
        min_pieces=min_pieces,
        max_pieces=max_pieces,
    )


# The shared reset helper retains success-gated retry and distractor behavior.
# This process-local replacement changes only how its Sobol move is selected.
chess_events.sample_chess_move_sobol = sample_indexed_piece_sobol_move


def log_status(level: int, message: str, *args) -> None:
    """Log and print operator-facing status messages.

    Isaac Sim/Kit can reconfigure Python logging after app startup, so mirror
    these sparse control-flow messages to stdout as well.
    """
    logger.log(level, message, *args)
    if args:
        message = message % args
    print(message, flush=True)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


@dataclass
class OnlineSubtaskAnnotationState:
    """Per-episode online subtask annotation state."""

    eef_queues: dict[str, list[str]]
    device: torch.device | str
    default_signal_dwell: int = 3
    signal_dwells: dict[str, int] = field(default_factory=dict)
    latched_signals: dict[str, bool] = field(init=False)
    latched_signal_tensors: dict[str, torch.Tensor] = field(init=False)
    eef_queue_indices: dict[str, int] = field(init=False)
    consecutive_true_counts: dict[str, int] = field(init=False)
    consecutive_true_count_tensors: dict[str, torch.Tensor] = field(init=False)
    required_dwell_tensors: dict[str, torch.Tensor] = field(init=False)

    def __post_init__(self) -> None:
        signal_names = [signal for queue in self.eef_queues.values() for signal in queue]
        duplicate_names = sorted({signal for signal in signal_names if signal_names.count(signal) > 1})
        if duplicate_names:
            raise ValueError(
                "Subtask term signal names must be globally unique for standard annotated teleop. "
                f"Duplicates: {duplicate_names}"
            )
        if not signal_names:
            raise ValueError("No subtask term signals found in env.cfg.subtask_configs.")
        self.latched_signals = {signal: False for signal in signal_names}
        self.latched_signal_tensors = {
            signal: torch.zeros((1, 1), dtype=torch.bool, device=self.device) for signal in signal_names
        }
        self.eef_queue_indices = {eef_name: 0 for eef_name in self.eef_queues}
        self.consecutive_true_counts = {eef_name: 0 for eef_name in self.eef_queues}
        self.consecutive_true_count_tensors = {
            eef_name: torch.zeros((), dtype=torch.long, device=self.device) for eef_name in self.eef_queues
        }
        self.required_dwell_tensors = {
            signal: torch.tensor(self._required_dwell(signal), dtype=torch.long, device=self.device)
            for signal in signal_names
        }

    @classmethod
    def from_env(
        cls,
        env: ManagerBasedRLEnv,
        *,
        default_signal_dwell: int,
        signal_dwells: dict[str, int],
    ) -> "OnlineSubtaskAnnotationState":
        subtask_cfgs = getattr(getattr(env, "cfg", None), "subtask_configs", {})
        if not isinstance(subtask_cfgs, dict) or not subtask_cfgs:
            raise ValueError("Annotated teleop requires env.cfg.subtask_configs to define one queue per EEF.")

        eef_queues: dict[str, list[str]] = {}
        for eef_name, cfgs in subtask_cfgs.items():
            queue = [
                str(getattr(cfg, "subtask_term_signal"))
                for cfg in cfgs
                if getattr(cfg, "subtask_term_signal", None)
            ]
            if queue:
                eef_queues[str(eef_name)] = queue

        return cls(
            eef_queues=eef_queues,
            device=env.device,
            default_signal_dwell=default_signal_dwell,
            signal_dwells=signal_dwells,
        )

    def reset(self) -> None:
        for signal_name in self.latched_signals:
            self.latched_signals[signal_name] = False
            self.latched_signal_tensors[signal_name].fill_(False)
        for eef_name in self.eef_queue_indices:
            self.eef_queue_indices[eef_name] = 0
            self.consecutive_true_counts[eef_name] = 0
            self.consecutive_true_count_tensors[eef_name].zero_()

    def current_signal_heads(self) -> dict[str, str | None]:
        return {eef_name: self._head_signal_for_eef(eef_name) for eef_name in self.eef_queues}

    def is_complete(self) -> bool:
        return all(
            int(self.eef_queue_indices.get(eef_name, 0)) >= len(queue)
            for eef_name, queue in self.eef_queues.items()
        )

    def as_tensor_dict(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            num_envs = 1
        elif isinstance(env_ids, torch.Tensor):
            num_envs = int(env_ids.numel())
        elif isinstance(env_ids, slice):
            num_envs = 1
        else:
            num_envs = len(env_ids)
        values = {}
        for signal_name, tensor in self.latched_signal_tensors.items():
            values[signal_name] = tensor if num_envs == 1 else tensor.expand(num_envs, 1)
        return values

    def advance(self, raw_signal_reader: Callable[[Sequence[str]], dict[str, Any]]) -> list[str]:
        active_heads = [
            (eef_name, signal_name)
            for eef_name, signal_name in self.current_signal_heads().items()
            if signal_name is not None
        ]
        if not active_heads:
            for eef_name in self.consecutive_true_count_tensors:
                self.consecutive_true_counts[eef_name] = 0
                self.consecutive_true_count_tensors[eef_name].zero_()
            return []

        active_signal_names = [signal_name for _, signal_name in active_heads]
        raw_signals = raw_signal_reader(active_signal_names)
        newly_latched: list[str] = []

        raw_true_values = []
        for eef_name, signal_name in active_heads:
            if signal_name not in raw_signals:
                available = sorted(raw_signals.keys())
                raise KeyError(
                    f"Raw subtask predicates did not include queue head '{signal_name}' for EEF '{eef_name}'. "
                    f"Available signals: {available}"
                )
            raw_true_values.append(
                torch.as_tensor(raw_signals[signal_name], device=self.device).reshape(-1)[0].to(dtype=torch.bool)
            )

        raw_true_tensor = torch.stack(raw_true_values)
        current_counts = torch.stack([self.consecutive_true_count_tensors[eef_name] for eef_name, _ in active_heads])
        updated_counts = torch.where(raw_true_tensor, current_counts + 1, torch.zeros_like(current_counts))
        required_dwells = torch.stack([self.required_dwell_tensors[signal_name] for _, signal_name in active_heads])
        ready_mask = updated_counts >= required_dwells

        for index, (eef_name, _) in enumerate(active_heads):
            self.consecutive_true_count_tensors[eef_name].copy_(updated_counts[index])

        count_and_ready = torch.stack((updated_counts, ready_mask.to(updated_counts.dtype)), dim=-1).cpu().tolist()
        for (eef_name, signal_name), (count_value, ready_value) in zip(active_heads, count_and_ready):
            self.consecutive_true_counts[eef_name] = int(count_value)
            if not bool(ready_value):
                continue

            self.latched_signals[signal_name] = True
            self.latched_signal_tensors[signal_name].fill_(True)
            self.eef_queue_indices[eef_name] += 1
            self.consecutive_true_counts[eef_name] = 0
            self.consecutive_true_count_tensors[eef_name].zero_()
            newly_latched.append(signal_name)

        return newly_latched

    def _head_signal_for_eef(self, eef_name: str) -> str | None:
        queue = self.eef_queues.get(eef_name, [])
        index = int(self.eef_queue_indices.get(eef_name, 0))
        if index >= len(queue):
            return None
        return queue[index]

    def _required_dwell(self, signal_name: str) -> int:
        return max(1, int(self.signal_dwells.get(signal_name, self.default_signal_dwell)))


def parse_signal_dwells(entries: Sequence[str]) -> dict[str, int]:
    signal_dwells: dict[str, int] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected --signal_dwell entries as SIGNAL=STEPS, got: {entry!r}")
        signal_name, dwell_text = entry.split("=", maxsplit=1)
        signal_name = signal_name.strip()
        if not signal_name:
            raise ValueError(f"Signal name cannot be empty in --signal_dwell entry: {entry!r}")
        dwell = int(dwell_text)
        if dwell < 1:
            raise ValueError(f"Signal dwell must be >= 1 for {signal_name!r}, got {dwell}.")
        signal_dwells[signal_name] = dwell
    return signal_dwells


def setup_output_directories() -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file) or "."
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, output_file_name


def create_environment_config(output_dir: str, output_file_name: str) -> ManagerBasedRLEnvCfg:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Annotated teleop recording is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}."
        )

    if args_cli.teleop_device.lower() == "so101leader":
        use_teleop_device = getattr(env_cfg, "use_teleop_device", None)
        if not callable(use_teleop_device):
            raise ValueError(
                f"Task {args_cli.task!r} does not provide use_teleop_device() for SO-101 leader control."
            )
        use_teleop_device("so101leader")

    env_cfg.env_name = args_cli.task.split(":")[-1]
    env_cfg.sim.log_dir = os.path.join(output_dir, ".isaaclab_logs")
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if hasattr(env_cfg.terminations, "failed"):
        # During human demonstration collection, subtask predicates annotate
        # progress but must not reject or reset the operator's attempt.
        env_cfg.terminations.failed = None

    chess_move_reset = getattr(env_cfg.events, "reset_sobol_chess_move", None)
    if chess_move_reset is not None:
        # Cycle through normalized types while retaining success-gated retry.
        # Geometry within each type comes from its own indexed Sobol stream.
        chess_move_reset.func = sample_type_balanced_chess_move_reset
        chess_move_reset.params["sampling_strategy"] = "sobol"
        chess_move_reset.params["advance_on_success_only"] = True
        if RECORDING_MOVE_PLAN:
            chess_move_reset.params["seed"] = 0
        else:
            chess_move_reset.params["seed"] = (
                int(args_cli.sobol_start_index) * len(RECORDING_PIECE_TYPES)
            )

    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.concatenate_terms = False

    if not args_cli.enable_cameras:
        env_cfg = remove_camera_configs(env_cfg)

    if args_cli.xr:
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    default_recorder_cfg = StandardAnnotatedMimicRecorderManagerCfg()
    if not isinstance(env_cfg.recorders, StandardAnnotatedMimicRecorderManagerCfg):
        env_cfg.recorders = default_recorder_cfg
    elif getattr(env_cfg.recorders, "record_pre_step_subtask_term_signals", None) is None:
        env_cfg.recorders.record_pre_step_subtask_term_signals = (
            default_recorder_cfg.record_pre_step_subtask_term_signals
        )
    env_cfg.recorders.record_pre_step_subtask_start_signals = None
    if not args_cli.enable_cameras:
        env_cfg.recorders.camera_names = []
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg


def setup_teleop_device(
    env: ManagerBasedRLEnv,
    env_cfg: ManagerBasedRLEnvCfg,
    callbacks: dict[str, Callable],
) -> object:
    if args_cli.teleop_device.lower() == "so101leader":
        from leisaac.devices import SO101Leader

        teleop_interface = SO101Leader(
            env,
            port=args_cli.port,
            recalibrate=args_cli.recalibrate,
            calibration_file=args_cli.calibration_file,
        )
        for key, callback in callbacks.items():
            teleop_interface.add_callback(key, callback)
        return teleop_interface

    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
        return create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)

    sensitivity = args_cli.sensitivity
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
        )
    else:
        raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")

    for key, callback in callbacks.items():
        try:
            teleop_interface.add_callback(key, callback)
        except (ValueError, TypeError) as exc:
            logger.warning("Failed to add callback for key %s: %s", key, exc)

    return teleop_interface


def is_unimplemented_mimic_method(env: ManagerBasedRLEnv, method_name: str) -> bool:
    method = getattr(env, method_name, None)
    method_func = getattr(method, "__func__", None)
    base_method = getattr(ManagerBasedRLMimicEnv, method_name, None)
    return method_func is not None and method_func is base_method


def validate_env_contract(env: ManagerBasedRLEnv) -> None:
    required_methods = ["get_robot_eef_pose", "get_object_poses", "action_to_target_eef_pose"]
    missing_methods = [method_name for method_name in required_methods if not hasattr(env, method_name)]
    if missing_methods:
        raise TypeError(
            "Annotated teleop requires a manager-based env with standard Mimic recording APIs. "
            f"Missing methods: {missing_methods}."
        )
    unimplemented_methods = [
        method_name
        for method_name in ("get_robot_eef_pose", "action_to_target_eef_pose")
        if is_unimplemented_mimic_method(env, method_name)
    ]
    if unimplemented_methods:
        raise NotImplementedError(
            "Annotated teleop requires environment-specific Mimic API implementations. "
            f"Unimplemented methods: {unimplemented_methods}."
        )
    has_raw_predicates = hasattr(env, "get_subtask_term_predicates")
    has_term_signals = hasattr(env, "get_subtask_term_signals") and not is_unimplemented_mimic_method(
        env, "get_subtask_term_signals"
    )
    if not has_raw_predicates and not has_term_signals:
        raise NotImplementedError(
            "Annotated teleop requires raw subtask predicates from get_subtask_term_predicates(env_ids=None), "
            "or an existing get_subtask_term_signals(env_ids=None) implementation to use as the raw source."
        )


def install_standard_mimic_method_adapters(env: ManagerBasedRLEnv) -> None:
    if not is_unimplemented_mimic_method(env, "get_object_poses"):
        return

    base_get_object_poses = env.get_object_poses

    def get_object_poses(_env, env_ids: Sequence[int] | None = None):
        return base_get_object_poses(env_ids=env_ids)

    env.get_object_poses = types.MethodType(get_object_poses, env)


def method_accepts_signal_names(method: Callable) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return "signal_names" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )


def make_raw_signal_reader(env: ManagerBasedRLEnv) -> Callable[[Sequence[str]], dict[str, Any]]:
    if hasattr(env, "get_subtask_term_predicates"):
        raw_signal_method = env.get_subtask_term_predicates
    else:
        raw_signal_method = env.get_subtask_term_signals

    accepts_signal_names = method_accepts_signal_names(raw_signal_method)

    def read_raw_signals(signal_names: Sequence[str]) -> dict[str, Any]:
        if accepts_signal_names:
            return raw_signal_method(env_ids=[0], signal_names=list(signal_names))
        return raw_signal_method(env_ids=[0])

    return read_raw_signals


def install_latched_signal_adapter(env: ManagerBasedRLEnv, annotator: OnlineSubtaskAnnotationState) -> None:
    def get_latched_subtask_term_signals(
        _env, env_ids: Sequence[int] | None = None, signal_names: Sequence[str] | None = None
    ):
        signals = annotator.as_tensor_dict(env_ids=env_ids)
        if signal_names is not None:
            return {signal_name: signals[signal_name] for signal_name in signal_names if signal_name in signals}
        return signals

    env.get_subtask_term_signals = types.MethodType(get_latched_subtask_term_signals, env)


def format_progress(annotator: OnlineSubtaskAnnotationState) -> str:
    parts = []
    for eef_name, queue in annotator.eef_queues.items():
        index = int(annotator.eef_queue_indices.get(eef_name, 0))
        head = annotator.current_signal_heads().get(eef_name)
        next_label = head if head is not None else "complete"
        parts.append(f"{eef_name}: {index}/{len(queue)} next={next_label}")
    return " | ".join(parts)


def reset_episode(env: ManagerBasedRLEnv, teleop_interface: object, annotator: OnlineSubtaskAnnotationState) -> None:
    env.recorder_manager.reset()
    annotator.reset()
    env.reset()
    destination_square = getattr(env, "active_target_square", None)
    if not isinstance(destination_square, (tuple, list)) or len(destination_square) != 2:
        raise RuntimeError("Chess reset did not select a destination square for recording.")
    row, column = (int(coordinate) for coordinate in destination_square)
    if not (0 <= row < 8 and 0 <= column < 8):
        raise ValueError(f"Invalid destination square selected at reset: {destination_square!r}")
    # The virtual destination pose is robot-root-relative in get_object_poses;
    # initial_state rigid-object poses are relative to the environment origin.
    destination_pose_b = env.get_object_poses(env_ids=[0])["destination_square"]
    robot = env.scene["robot"]
    robot_pose_env = make_pose(
        robot.data.root_pos_w[:1] - env.scene.env_origins[:1],
        matrix_from_quat(robot.data.root_quat_w[:1]),
    )
    destination_pose_env = robot_pose_env @ destination_pose_b
    env.recorder_manager.add_to_episodes(
        "initial_state/rigid_objects/destination_square",
        {
            "initial_pose": destination_pose_env,
            "scale": torch.ones((1, 3), dtype=destination_pose_env.dtype, device=env.device),
        },
        env_ids=[0],
    )
    teleop_interface.reset()


def normalize_action(env: ManagerBasedRLEnv, action: Any) -> torch.Tensor | None:
    if action is None:
        return None
    action_tensor = torch.as_tensor(action, device=env.device, dtype=torch.float32)
    if action_tensor.ndim == 1:
        action_tensor = action_tensor.unsqueeze(0)
    return action_tensor.repeat(env.num_envs, 1)


def export_successful_episode(env: ManagerBasedRLEnv) -> None:
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
    env.recorder_manager.export_episodes([0])


def main() -> int:
    if RECORDING_MOVE_PLAN and args_cli.num_demos != len(RECORDING_MOVE_PLAN):
        raise ValueError(
            "--num_demos must equal the number of --sobol_plan entries: "
            f"{args_cli.num_demos} != {len(RECORDING_MOVE_PLAN)}"
        )
    if args_cli.default_signal_dwell < 1:
        raise ValueError(f"--default_signal_dwell must be >= 1, got {args_cli.default_signal_dwell}.")
    if args_cli.num_demos > 0 and args_cli.num_demos % len(RECORDING_PIECE_TYPES) != 0:
        log_status(
            logging.WARNING,
            "--num_demos=%d is not divisible by %d; the final partial cycle will not contain every piece type.",
            args_cli.num_demos,
            len(RECORDING_PIECE_TYPES),
        )

    output_dir, output_file_name = setup_output_directories()
    env_cfg = create_environment_config(output_dir, output_file_name)
    env: ManagerBasedRLEnv | None = None
    teleop_interface: object | None = None

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        if not isinstance(env, ManagerBasedRLEnv):
            raise ValueError(
                "Annotated teleop recording requires a ManagerBasedRLEnv environment. "
                f"Received: {type(env).__name__}."
            )
        validate_env_contract(env)
        install_standard_mimic_method_adapters(env)

        runtime_step_hz = 1.0 / env.step_dt
        if args_cli.step_hz > 0 and not math.isclose(
            runtime_step_hz, args_cli.step_hz, rel_tol=0.0, abs_tol=1.0e-7
        ):
            raise RuntimeError(
                f"Task control rate is {runtime_step_hz:.9f} Hz, but --step_hz={args_cli.step_hz}. "
                "The flag only paces wall time and cannot resample the dataset."
        )
        log_status(logging.INFO, "Dataset sampling rate: %.3f Hz.", runtime_step_hz)
        log_status(
            logging.INFO,
            "Sobol recording piece types: %s; target=%d successful demonstrations.",
            ", ".join(RECORDING_PIECE_TYPES),
            args_cli.num_demos,
        )

        signal_dwells = parse_signal_dwells(args_cli.signal_dwell)
        annotator = OnlineSubtaskAnnotationState.from_env(
            env,
            default_signal_dwell=args_cli.default_signal_dwell,
            signal_dwells=signal_dwells,
        )
        raw_signal_reader = make_raw_signal_reader(env)
        install_latched_signal_adapter(env, annotator)
        log_status(logging.INFO, "Annotation queues configured: %s", format_progress(annotator))

        recorded_demo_count = 0
        recording_active = False
        completion_announced = False
        flags = {"start": False, "save": False, "discard": False, "abort": False, "reset": False}

        def on_start() -> None:
            flags["start"] = True
            log_status(logging.INFO, "[START] Recording start requested.")

        def on_save() -> None:
            flags["save"] = True
            log_status(logging.INFO, "[SAVE] Save requested.")

        def on_discard() -> None:
            flags["discard"] = True
            log_status(logging.INFO, "[DISCARD] Discard requested.")

        def on_abort() -> None:
            flags["abort"] = True
            log_status(logging.WARNING, "[ESC] Abort requested.")

        def on_reset() -> None:
            if recording_active and annotator.is_complete():
                flags["save"] = True
                log_status(logging.INFO, "[RESET] Save requested because annotation queues are complete.")
                return
            flags["reset"] = True
            log_status(logging.INFO, "[RESET] Reset requested.")

        callbacks = {
            "S": on_start,
            "N": on_save,
            "D": on_discard,
            "R": on_discard,
            "RESET": on_reset,
            "START": on_start,
            "STOP": on_save,
            "ESCAPE": on_abort,
        }
        log_status(logging.INFO, "Creating teleop device: %s", args_cli.teleop_device)
        teleop_interface = setup_teleop_device(env, env_cfg, callbacks)
        if hasattr(teleop_interface, "display_controls"):
            teleop_interface.display_controls()

        rate_limiter = RateLimiter(args_cli.step_hz) if args_cli.step_hz > 0 else None
        log_status(logging.INFO, "Resetting annotated recording episode.")
        reset_episode(env, teleop_interface, annotator)

        log_status(logging.INFO, "Using teleop device: %s", teleop_interface)
        if args_cli.xr:
            log_status(
                logging.INFO,
                "XR controls: START begins recording, RESET saves after queues complete or discards before completion.",
            )
        elif args_cli.teleop_device.lower() == "so101leader":
            log_status(
                logging.INFO,
                "Press B to start, N to save after all queues complete, R to discard/reset, Ctrl+C to abort.",
            )
        else:
            log_status(
                logging.INFO,
                "Press S to start, N to save after all queues complete, D/R to discard and reset, ESC to abort.",
            )
        log_status(logging.INFO, "Annotation queues: %s", format_progress(annotator))

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                if flags["abort"]:
                    break

                if flags["reset"] or flags["discard"]:
                    reset_episode(env, teleop_interface, annotator)
                    recording_active = False
                    completion_announced = False
                    flags["reset"] = False
                    flags["discard"] = False
                    flags["save"] = False
                    log_status(logging.INFO, "Episode discarded. Start a new attempt when ready.")
                    continue

                if flags["start"] and not recording_active:
                    recording_active = True
                    completion_announced = False
                    flags["start"] = False
                    log_status(logging.INFO, "Recording active. %s", format_progress(annotator))

                # LeIsaac's physical leader handles B internally instead of
                # emitting Isaac Lab's generic START callback.
                if (
                    not recording_active
                    and args_cli.teleop_device.lower() == "so101leader"
                    and bool(getattr(teleop_interface, "started", False))
                ):
                    recording_active = True
                    completion_announced = False
                    log_status(logging.INFO, "Recording active. %s", format_progress(annotator))

                if flags["save"]:
                    if recording_active and annotator.is_complete():
                        export_successful_episode(env)
                        # Mark this attempt consumed before a possible loop
                        # exit. Otherwise the post-loop safety save would
                        # export the final episode a second time.
                        recording_active = False
                        recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                        log_status(
                            logging.INFO,
                            "Saved annotated episode. Recorded %d successful demonstrations.",
                            recorded_demo_count,
                        )
                        if args_cli.num_demos > 0 and recorded_demo_count >= args_cli.num_demos:
                            break
                        reset_episode(env, teleop_interface, annotator)
                        completion_announced = False
                        log_status(logging.INFO, "Ready for next episode. Start when ready.")
                    else:
                        log_status(
                            logging.WARNING,
                            "Save ignored because annotation queues are not complete. %s",
                            format_progress(annotator),
                        )
                    flags["save"] = False
                    continue

                if not recording_active:
                    env.sim.render()
                    if rate_limiter is not None:
                        rate_limiter.sleep(env)
                    continue

                try:
                    action = normalize_action(env, teleop_interface.advance())
                except Exception as exc:
                    logger.error("Error in teleop interface: %s", exc, exc_info=True)
                    action = None

                if action is None:
                    env.sim.render()
                    if rate_limiter is not None:
                        rate_limiter.sleep(env)
                    continue

                newly_latched = annotator.advance(raw_signal_reader)
                if newly_latched:
                    log_status(logging.INFO, "Latched: %s", ", ".join(newly_latched))
                    log_status(logging.INFO, "Annotation progress: %s", format_progress(annotator))

                # Publish the current queue heads so subtask observation
                # functions can gate any per-step debug printing to the
                # signal each EEF is actively dwelling on.
                env._debug_subtask_heads = {
                    signal for signal in annotator.current_signal_heads().values()
                    if signal is not None
                }

                _, _, terminated, truncated, _ = env.step(action)
                episode_ended = bool(torch.any(terminated).item() or torch.any(truncated).item())

                if annotator.is_complete() and not completion_announced and not episode_ended:
                    if args_cli.xr:
                        log_status(
                            logging.INFO,
                            "All annotation queues completed. Send RESET to save the episode.",
                        )
                    elif args_cli.teleop_device.lower() == "so101leader":
                        log_status(
                            logging.INFO,
                            "All annotation queues completed. Press N to save the episode or R to re-record.",
                        )
                    else:
                        log_status(
                            logging.INFO,
                            "All annotation queues completed. Press N to save the episode or D to re-record.",
                        )
                    completion_announced = True

                if episode_ended:
                    failed_subtasks = [
                        signal_name
                        for signal_name in annotator.current_signal_heads().values()
                        if signal_name is not None
                    ]
                    failure_context = ", ".join(failed_subtasks) if failed_subtasks else "unknown subtask"
                    log_status(
                        logging.WARNING,
                        "Episode failed during %s and was discarded. Ready for the next episode.",
                        failure_context,
                    )
                    # ManagerBasedRLEnv.step() already reset the terminated env,
                    # including the recorder and Sobol move sampler. Reset only
                    # operator-side state so one failure advances exactly once.
                    annotator.reset()
                    teleop_interface.reset()
                    recording_active = False
                    completion_announced = False

                if rate_limiter is not None:
                    rate_limiter.sleep(env)

        if recording_active and annotator.is_complete():
            log_status(logging.INFO, "Saving completed episode before exit.")
            export_successful_episode(env)
            recorded_demo_count = env.recorder_manager.exported_successful_episode_count
            log_status(
                logging.INFO,
                "Saved annotated episode. Recorded %d successful demonstrations.",
                recorded_demo_count,
            )

        return recorded_demo_count
    finally:
        if teleop_interface is not None and hasattr(teleop_interface, "is_connected"):
            try:
                if bool(teleop_interface.is_connected):
                    teleop_interface.disconnect()
            except Exception as exc:
                logger.warning("Failed to disconnect teleop device cleanly: %s", exc)
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        # SimulationApp.close() can otherwise mask startup/configuration
        # failures by terminating the process with a successful exit code.
        os._exit(1)
    finally:
        simulation_app.close()
