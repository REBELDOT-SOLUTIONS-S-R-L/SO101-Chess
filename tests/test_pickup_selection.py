from types import SimpleNamespace

import torch

from so101_chess.tasks.manager_based.so101_chess.pickup_selection import (
    TypeFilteredMultiObjectStrategy,
    TypeFilteredPickupStrategy,
)


def _pose(x: float = 0.0) -> torch.Tensor:
    pose = torch.eye(4)
    pose[0, 3] = x
    return pose


def _datagen_info(x: float = 0.0):
    pose = _pose(x).unsqueeze(0)
    return SimpleNamespace(eef_pose=pose, object_poses={"active_piece": pose})


def test_pickup_exclusion_removes_fragile_nearest_source():
    infos = [_datagen_info(0.0), _datagen_info(0.1)]
    selected = TypeFilteredPickupStrategy().select_source_demo(
        eef_pose=_pose(0.0),
        object_pose=_pose(0.0),
        src_subtask_datagen_infos=infos,
        current_piece_type="rook_white",
        source_piece_types=["rook", "rook"],
        current_joint_positions=torch.zeros(5),
        source_joint_positions=[torch.zeros(5), torch.ones(5)],
        excluded_demo_indices=[0],
        nn_k=1,
    )

    assert selected == 1


def test_multi_object_selection_filters_candidates_by_piece_type():
    infos = [_datagen_info(), _datagen_info(), _datagen_info()]
    source_poses = [
        {"active_piece": _pose(0.0), "destination_square": _pose(0.0)},
        {"active_piece": _pose(0.5), "destination_square": _pose(0.5)},
        {"active_piece": _pose(0.0), "destination_square": _pose(0.0)},
    ]
    selected = TypeFilteredMultiObjectStrategy().select_source_demo(
        eef_pose=_pose(),
        object_pose=_pose(),
        src_subtask_datagen_infos=infos,
        current_piece_type="queen_black",
        source_piece_types=["pawn", "queen", "rook"],
        all_object_poses={"active_piece": _pose(), "destination_square": _pose()},
        src_all_object_poses=source_poses,
        object_names=["active_piece", "destination_square"],
        rot_weight=0.0,
        nn_k=1,
    )

    assert selected == 1
