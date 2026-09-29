import importlib.util
from dataclasses import dataclass
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "source/so101_chess/so101_chess/tasks/manager_based/so101_chess/pickup_trajectory.py"
)
SPEC = importlib.util.spec_from_file_location("pickup_trajectory", MODULE_PATH)
pickup_trajectory = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pickup_trajectory)


@dataclass
class FakeWaypoint:
    pose: torch.Tensor
    gripper_action: torch.Tensor


@dataclass
class FakeSequence:
    sequence: list[FakeWaypoint]


@dataclass
class FakeTrajectory:
    waypoint_sequences: list[FakeSequence]


def _trajectory() -> FakeTrajectory:
    waypoints = [
        FakeWaypoint(torch.eye(4), torch.tensor([value]))
        for value in (0.62, 0.31, -0.17)
    ]
    return FakeTrajectory([FakeSequence(waypoints)])


def _gripper_commands(trajectory: FakeTrajectory) -> list[torch.Tensor]:
    return [
        waypoint.gripper_action.clone()
        for sequence in trajectory.waypoint_sequences
        for waypoint in sequence.sequence
    ]


def test_tcp_offset_preserves_source_gripper_commands():
    trajectory = _trajectory()
    source_commands = _gripper_commands(trajectory)

    pickup_trajectory.adjust_pickup_waypoint_poses(
        trajectory,
        fixed_offset=torch.tensor([0.01, -0.02, 0.03]),
    )

    for actual, expected in zip(_gripper_commands(trajectory), source_commands, strict=True):
        torch.testing.assert_close(actual, expected)
    for waypoint in trajectory.waypoint_sequences[0].sequence:
        torch.testing.assert_close(waypoint.pose[:3, 3], torch.tensor([0.01, -0.02, 0.03]))


def test_grasp_pose_hold_preserves_source_gripper_commands():
    trajectory = _trajectory()
    source_commands = _gripper_commands(trajectory)
    held_pose = torch.eye(4)
    held_pose[:3, 3] = torch.tensor([0.2, 0.1, 0.04])

    pickup_trajectory.adjust_pickup_waypoint_poses(trajectory, held_pose=held_pose)

    for actual, expected in zip(_gripper_commands(trajectory), source_commands, strict=True):
        torch.testing.assert_close(actual, expected)
    for waypoint in trajectory.waypoint_sequences[0].sequence:
        torch.testing.assert_close(waypoint.pose, held_pose)


def test_measured_completion_repeats_final_waypoint_until_latched():
    state = {"latched": False}
    first = object()
    final = object()
    waypoints = pickup_trajectory.MeasuredCompletionWaypointList(
        [first, final],
        completion_predicate=lambda: state["latched"],
    )

    assert waypoints[0] is first
    assert waypoints[1] is final
    next_index = 2
    assert next_index != len(waypoints)

    assert waypoints[next_index] is final
    state["latched"] = True
    next_index += 1
    assert next_index == len(waypoints)
    assert list.__len__(waypoints) == 2


def test_measured_completion_still_executes_full_source_segment():
    waypoints = pickup_trajectory.MeasuredCompletionWaypointList(
        ["first", "middle", "final"],
        completion_predicate=lambda: True,
    )

    assert waypoints[0] == "first"
    assert len(waypoints) == 3
    assert waypoints[1] == "middle"
    assert len(waypoints) == 3
    assert waypoints[2] == "final"
    assert len(waypoints) == 3


def test_trailing_closing_commands_move_from_pregrasp_to_grasp():
    boundaries = [(0, 2), (2, 7), (7, 10), (10, 12)]
    gripper_actions = torch.tensor(
        [-0.17, 0.40, 0.42, 0.42, 0.31, 0.24, 0.15, 0.08, -0.05, -0.17, -0.17, -0.17]
    )

    adjusted = pickup_trajectory.move_trailing_closing_commands_to_grasp(
        boundaries,
        gripper_actions,
        pregrasp_index=1,
        grasp_index=2,
        open_threshold=0.3,
    )

    assert adjusted == [(0, 2), (2, 5), (5, 10), (10, 12)]
    assert boundaries == [(0, 2), (2, 7), (7, 10), (10, 12)]
    assert gripper_actions[adjusted[1][1] - 1] >= 0.3
    assert gripper_actions[adjusted[2][0]] < 0.3


def test_open_pregrasp_boundary_is_unchanged():
    boundaries = [(0, 2), (2, 5), (5, 8)]
    gripper_actions = torch.tensor([-0.17, 0.40, 0.45, 0.42, 0.31, 0.20, 0.0, -0.17])

    adjusted = pickup_trajectory.move_trailing_closing_commands_to_grasp(
        boundaries,
        gripper_actions,
        pregrasp_index=1,
        grasp_index=2,
        open_threshold=0.3,
    )

    assert adjusted == boundaries


def test_pickup_z_clamp_only_raises_waypoints_below_floor():
    trajectory = _trajectory()
    source_z = (0.010, 0.024, 0.060)
    for waypoint, z in zip(trajectory.waypoint_sequences[0].sequence, source_z, strict=True):
        waypoint.pose[2, 3] = z

    pickup_trajectory.clamp_pickup_waypoint_z(trajectory, 0.024)

    actual_z = torch.stack(
        [waypoint.pose[2, 3] for waypoint in trajectory.waypoint_sequences[0].sequence]
    )
    torch.testing.assert_close(actual_z, torch.tensor([0.024, 0.024, 0.060]))


def test_pickup_z_clamp_preserves_gripper_commands():
    trajectory = _trajectory()
    source_commands = _gripper_commands(trajectory)

    pickup_trajectory.clamp_pickup_waypoint_z(trajectory, torch.tensor(0.040))

    for actual, expected in zip(_gripper_commands(trajectory), source_commands, strict=True):
        torch.testing.assert_close(actual, expected)
