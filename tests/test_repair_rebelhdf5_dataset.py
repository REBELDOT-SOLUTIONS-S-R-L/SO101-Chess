import importlib.util
from pathlib import Path

import h5py
import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "repair_rebelhdf5_dataset.py"
SPEC = importlib.util.spec_from_file_location("repair_rebelhdf5_dataset", SCRIPT_PATH)
repair_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(repair_module)


def _pose(x: float, y: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float32)[None]
    value[0, 0, 3] = x
    value[0, 1, 3] = y
    return value


def test_repair_keeps_only_active_piece_and_adds_source_alias(tmp_path):
    dataset_path = tmp_path / "dataset.hdf5"
    with h5py.File(dataset_path, "w") as dataset:
        demo = dataset.create_group("data/demo_0")
        rigid_objects = demo.create_group("initial_state/rigid_objects")
        rigid_objects.create_group("bishop_white").create_dataset(
            "initial_pose", data=_pose(0.1, -0.1)
        )
        rigid_objects.create_group("rook_white").create_dataset(
            "initial_pose", data=_pose(10.0, 10.0)
        )
        references = demo.create_group("reference_demo_indices")
        references.create_dataset("chess", data=np.array([4, 2, 9], dtype=np.int64))

    result = repair_module.repair_file(dataset_path)

    assert result == (1, 1)
    with h5py.File(dataset_path, "r") as dataset:
        demo = dataset["data/demo_0"]
        assert list(demo["initial_state/rigid_objects"].keys()) == ["active_piece"]
        np.testing.assert_array_equal(
            demo["reference_demo_indices/left"][...],
            demo["reference_demo_indices/chess"][...],
        )


def test_repair_is_idempotent(tmp_path):
    dataset_path = tmp_path / "dataset.hdf5"
    with h5py.File(dataset_path, "w") as dataset:
        demo = dataset.create_group("data/demo_0")
        active_piece = demo.create_group("initial_state/rigid_objects/active_piece")
        active_piece.create_dataset("initial_pose", data=_pose(0.0, 0.0))
        references = demo.create_group("reference_demo_indices")
        canonical = references.create_dataset("chess", data=np.array([1], dtype=np.int64))
        references["left"] = canonical

    assert repair_module.repair_file(dataset_path) == (1, 0)
    assert repair_module.repair_file(dataset_path) == (1, 0)


def test_repair_normalizes_legacy_initial_and_per_step_states(tmp_path):
    dataset_path = tmp_path / "legacy_dataset.hdf5"
    active_root_pose = np.array([[0.1, -0.1, 0.7, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    parked_root_pose = np.array([[10.0, 10.0, 0.8, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    with h5py.File(dataset_path, "w") as dataset:
        demo = dataset.create_group("data/demo_0")
        initial_objects = demo.create_group("initial_state/rigid_object")
        initial_objects.create_group("bishop_white").create_dataset(
            "root_pose", data=active_root_pose
        )
        initial_objects.create_group("rook_white").create_dataset(
            "root_pose", data=parked_root_pose
        )
        state_objects = demo.create_group("states/rigid_object")
        state_objects.create_group("bishop_white").create_dataset(
            "root_pose", data=np.repeat(active_root_pose, 3, axis=0)
        )
        state_objects.create_group("rook_white").create_dataset(
            "root_pose", data=np.repeat(parked_root_pose, 3, axis=0)
        )

    assert repair_module.repair_file(dataset_path) == (1, 0)
    with h5py.File(dataset_path, "r") as dataset:
        demo = dataset["data/demo_0"]
        assert list(demo["initial_state/rigid_object"]) == ["active_piece"]
        assert list(demo["states/rigid_object"]) == ["active_piece"]
