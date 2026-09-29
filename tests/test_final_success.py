from types import SimpleNamespace

import torch

from so101_chess.tasks.manager_based.so101_chess.mdp import terminations


class _FakeEnv:
    def __init__(self):
        class _FakeScene(dict):
            pass

        piece_data = SimpleNamespace(
            root_pos_w=torch.tensor([[0.20, 0.10, 0.02]], dtype=torch.float32),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
        self.scene = _FakeScene(active_piece=SimpleNamespace(data=piece_data))
        self.scene.env_origins = torch.zeros((1, 3), dtype=torch.float32)
        self.num_envs = 1
        self.device = "cpu"
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        # These would reject an episode under the generation-time predicates.
        self._chess_subtask_stage = torch.zeros(1, dtype=torch.long)
        self._chess_subtask_failed = torch.ones(1, dtype=torch.bool)


def test_final_success_ignores_subtask_stage_failure_and_gripper(monkeypatch):
    env = _FakeEnv()
    monkeypatch.setattr(terminations, "_get_active_piece_name", lambda _env: "active_piece")
    monkeypatch.setattr(
        terminations,
        "_source_target_world",
        lambda _env, _name: torch.tensor([0.20, 0.10, 0.02]),
    )
    monkeypatch.setattr(
        terminations,
        "_eef_position_robot_root",
        lambda _env, _link: torch.tensor([[0.05, -0.12, 0.0135]]),
    )

    for step in range(1, 5):
        env.episode_length_buf[0] = step
        assert not bool(terminations.chess_move_final_success(env)[0])

    env.episode_length_buf[0] = 5
    assert bool(terminations.chess_move_final_success(env)[0])


def test_final_success_requires_both_placement_and_home(monkeypatch):
    env = _FakeEnv()
    monkeypatch.setattr(terminations, "_get_active_piece_name", lambda _env: "active_piece")
    monkeypatch.setattr(
        terminations,
        "_source_target_world",
        lambda _env, _name: torch.tensor([0.20, 0.10, 0.02]),
    )
    monkeypatch.setattr(
        terminations,
        "_eef_position_robot_root",
        lambda _env, _link: torch.tensor([[0.20, -0.12, 0.0135]]),
    )

    for step in range(1, 8):
        env.episode_length_buf[0] = step
        assert not bool(terminations.chess_move_final_success(env)[0])


def test_reset_clears_final_success_hold_counter():
    env = _FakeEnv()
    env._chess_final_success_hold_counter = torch.tensor([4], dtype=torch.long)
    env._chess_final_success_last_counted_step = torch.tensor([123], dtype=torch.long)

    terminations.reset_chess_hold_counter(env, torch.tensor([0], dtype=torch.long))

    assert env._chess_final_success_hold_counter.tolist() == [0]
    assert env._chess_final_success_last_counted_step.tolist() == [-1]
