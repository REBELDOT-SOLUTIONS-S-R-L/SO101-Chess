from types import SimpleNamespace

import torch

from so101_chess.tasks.manager_based.so101_chess.mdp import domain_randomization


class _FakeCamera:
    device = "cpu"

    def __init__(self):
        self.cfg = SimpleNamespace(
            offset=SimpleNamespace(
                pos=(-0.15, 0.0, 1.20),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="opengl",
            )
        )
        self.positions = None
        self.orientations = None
        self.reset_count = 0

    def set_world_poses(self, positions, orientations, env_ids, convention):
        self.positions = positions.clone()
        self.orientations = orientations.clone()
        self.env_ids = env_ids.clone()
        self.convention = convention

    def reset(self, env_ids):
        self.reset_count += 1
        self.reset_env_ids = env_ids.clone()


class _FakeScene(dict):
    def __init__(self, camera):
        super().__init__(top_camera=camera)
        self.env_origins = torch.tensor([[2.0, -3.0, 0.0]])


class _FakeEnv:
    num_envs = 1
    device = "cpu"

    def __init__(self, camera):
        self.scene = _FakeScene(camera)


def test_camera_pose_randomization_skips_a_removed_camera():
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=SimpleNamespace(keys=lambda: [], env_origins=torch.zeros((1, 3))),
    )

    domain_randomization.randomize_fixed_camera_pose(
        env,
        torch.tensor([0]),
        position_range={"x": (-0.005, 0.005)},
    )


def test_camera_pose_remains_bounded_across_1000_resets():
    camera = _FakeCamera()
    env = _FakeEnv(camera)
    env_ids = torch.tensor([0], dtype=torch.long)
    nominal_world_position = env.scene.env_origins + torch.tensor(camera.cfg.offset.pos)
    position_limit = 0.005
    rotation_limit = torch.deg2rad(torch.tensor(0.5)).item()

    torch.manual_seed(7)
    for _ in range(1000):
        domain_randomization.randomize_fixed_camera_pose(
            env,
            env_ids,
            position_range={axis: (-position_limit, position_limit) for axis in ("x", "y", "z")},
            rotation_range={axis: (-rotation_limit, rotation_limit) for axis in ("roll", "pitch", "yaw")},
        )
        offset = camera.positions - nominal_world_position
        assert torch.all(offset >= -position_limit - 1e-7)
        assert torch.all(offset <= position_limit + 1e-7)
        assert torch.allclose(torch.linalg.vector_norm(camera.orientations, dim=1), torch.ones(1), atol=1e-6)

    assert camera.reset_count == 1000
    assert torch.equal(camera.env_ids, env_ids)
    assert camera.convention == "opengl"


def test_light_offsets_remain_bounded_across_1000_resets():
    position_limits = torch.tensor([1.50, 1.50, 1.00])
    rotation_limit = torch.deg2rad(torch.tensor(45.0)).item()
    position_range = {
        axis: (-float(position_limits[index]), float(position_limits[index]))
        for index, axis in enumerate(("x", "y", "z"))
    }
    rotation_range = {
        axis: (-rotation_limit, rotation_limit) for axis in ("roll", "pitch", "yaw")
    }

    torch.manual_seed(11)
    for _ in range(1000):
        position_offsets = domain_randomization._sample_axis_offsets(
            3, position_range, domain_randomization._POSITION_AXES, "cpu"
        )
        rotation_offsets = domain_randomization._sample_axis_offsets(
            3, rotation_range, domain_randomization._ROTATION_AXES, "cpu"
        )
        assert torch.all(position_offsets >= -position_limits)
        assert torch.all(position_offsets <= position_limits)
        assert torch.all(rotation_offsets >= -rotation_limit)
        assert torch.all(rotation_offsets <= rotation_limit)


def test_light_activation_always_keeps_the_configured_minimum():
    torch.manual_seed(13)
    for _ in range(1000):
        enabled = domain_randomization._sample_enabled_lights(
            count=3,
            off_probability=1.0,
            min_active_lights=1,
        )
        assert enabled.dtype == torch.bool
        assert int(enabled.sum().item()) == 1

    assert torch.all(
        domain_randomization._sample_enabled_lights(
            count=3,
            off_probability=0.0,
            min_active_lights=1,
        )
    )


def test_lights_never_move_below_the_minimum_height():
    minimum_height = 0.90

    assert domain_randomization._enforce_minimum_height(
        (1.0, -2.0, -5.0), minimum_height
    ) == (1.0, -2.0, minimum_height)
    assert domain_randomization._enforce_minimum_height(
        (1.0, -2.0, 1.75), minimum_height
    ) == (1.0, -2.0, 1.75)
