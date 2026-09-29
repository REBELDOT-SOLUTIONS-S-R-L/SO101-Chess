import torch

from so101_chess.tasks.manager_based.so101_chess.mdp import events as chess_events
from so101_chess.tasks.manager_based.so101_chess.piece_types import normalize_piece_type

sample_chess_move_sobol = chess_events.sample_chess_move_sobol


def test_sample_chess_move_sobol_returns_valid_move():
    piece_names = ["bishop_white", "queen_white", "pawn_white"]

    piece_name, source_square, target_square = sample_chess_move_sobol(
        seed=123,
        piece_names=piece_names,
    )

    assert piece_name in piece_names
    assert 0 <= source_square[0] <= 7
    assert 0 <= source_square[1] <= 7
    assert 0 <= target_square[0] <= 7
    assert 0 <= target_square[1] <= 7
    assert source_square != target_square


def test_sample_chess_move_random_returns_valid_move():
    piece_names = ["rook_white", "knight_white"]

    piece_name, source_square, target_square = chess_events.sample_chess_move_random(
        piece_names=piece_names,
    )

    assert piece_name in piece_names
    assert target_square in chess_events.get_valid_chess_targets(piece_name, source_square)


def test_type_balanced_random_cycles_piece_types_before_sampling_move():
    sampled_types = []
    for type_index in range(12):
        piece_name, source_square, target_square = chess_events.sample_chess_move_type_balanced_random(
            type_index=type_index,
        )
        sampled_types.append(normalize_piece_type(piece_name))
        assert target_square in chess_events.get_valid_chess_targets(piece_name, source_square)

    expected_cycle = list(chess_events.PIECE_TYPES)
    assert sampled_types == expected_cycle + expected_cycle


def test_chess_move_reset_advances_only_after_success(monkeypatch):
    class RecorderManager:
        exported_successful_episode_count = 0

    class Env:
        num_envs = 1
        device = "cpu"
        scene = {"pawn_white": object()}
        recorder_manager = RecorderManager()

    env = Env()
    env_ids = torch.tensor([0], dtype=torch.long)

    monkeypatch.setattr(chess_events, "move_piece_off_board", lambda *args, **kwargs: None)
    monkeypatch.setattr(chess_events, "place_piece_on_square", lambda *args, **kwargs: None)
    monkeypatch.setattr(chess_events, "_set_piece_awake", lambda *args, **kwargs: None)
    monkeypatch.setattr(chess_events, "color_board_squares", lambda *args, **kwargs: None)

    reset_kwargs = {
        "piece_names": ["pawn_white"],
        "seed": 123,
        "advance_on_success_only": True,
    }

    first_move = chess_events.sample_chess_move_reset(env, env_ids, **reset_kwargs)
    assert env._chess_move_reset_state["count"] == 1

    retried_move = chess_events.sample_chess_move_reset(env, env_ids, **reset_kwargs)
    assert retried_move == first_move
    assert env._chess_move_reset_state["count"] == 1

    env.recorder_manager.exported_successful_episode_count = 1
    chess_events.sample_chess_move_reset(env, env_ids, **reset_kwargs)
    assert env._chess_move_reset_state["count"] == 2

    env._chess_move_episode_succeeded = torch.tensor([True])
    chess_events.sample_chess_move_reset(env, env_ids, **reset_kwargs)
    assert env._chess_move_reset_state["count"] == 3
    assert not bool(env._chess_move_episode_succeeded[0])


def test_chess_move_reset_uses_fresh_random_move_on_every_reset(monkeypatch):
    class Env:
        num_envs = 1
        device = "cpu"
        scene = {"rook_white": object()}

    env = Env()
    env_ids = torch.tensor([0], dtype=torch.long)
    sampled_moves = [
        ("rook_white", (0, 0), (0, 1)),
        ("rook_white", (1, 0), (1, 1)),
    ]

    monkeypatch.setattr(chess_events, "move_piece_off_board", lambda *args, **kwargs: None)
    monkeypatch.setattr(chess_events, "place_piece_on_square", lambda *args, **kwargs: None)
    monkeypatch.setattr(chess_events, "_set_piece_awake", lambda *args, **kwargs: None)
    monkeypatch.setattr(chess_events, "color_board_squares", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        chess_events,
        "sample_chess_move_random",
        lambda **kwargs: sampled_moves.pop(0),
    )
    monkeypatch.setattr(
        chess_events,
        "sample_chess_move_sobol",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Sobol sampler was called")),
    )

    reset_kwargs = {
        "piece_names": ["rook_white"],
        "sampling_strategy": "random",
        "advance_on_success_only": False,
    }
    first_move = chess_events.sample_chess_move_reset(env, env_ids, **reset_kwargs)
    second_move = chess_events.sample_chess_move_reset(env, env_ids, **reset_kwargs)

    assert first_move == ("rook_white", (0, 0), (0, 1))
    assert second_move == ("rook_white", (1, 0), (1, 1))
    assert env._chess_move_reset_state["count"] == 2


def test_chess_move_reset_cycles_type_after_success_with_fresh_failed_moves(monkeypatch):
    class RecorderManager:
        exported_successful_episode_count = 0

    class Env:
        num_envs = 1
        device = "cpu"
        scene = {name: object() for name in chess_events._DEFAULT_CHESS_PIECES}
        recorder_manager = RecorderManager()

    env = Env()
    env_ids = torch.tensor([0], dtype=torch.long)
    sampled_type_indices = []

    monkeypatch.setattr(chess_events, "move_piece_off_board", lambda *args, **kwargs: None)
    monkeypatch.setattr(chess_events, "place_piece_on_square", lambda *args, **kwargs: None)
    monkeypatch.setattr(chess_events, "_set_piece_awake", lambda *args, **kwargs: None)
    monkeypatch.setattr(chess_events, "color_board_squares", lambda *args, **kwargs: None)

    def sample_balanced(**kwargs):
        sampled_type_indices.append(kwargs["type_index"])
        piece_type = chess_events.PIECE_TYPES[kwargs["type_index"] % len(chess_events.PIECE_TYPES)]
        return f"{piece_type}_white", (0, 0), chess_events.get_valid_chess_targets(
            f"{piece_type}_white", (0, 0)
        )[0]

    monkeypatch.setattr(chess_events, "sample_chess_move_type_balanced_random", sample_balanced)

    for attempt_index in range(12):
        if attempt_index > 0 and attempt_index % 2 == 0:
            env.recorder_manager.exported_successful_episode_count += 1
        chess_events.sample_chess_move_reset(
            env,
            env_ids,
            sampling_strategy="type_balanced_random",
            advance_on_success_only=False,
            min_pieces=1,
            max_pieces=1,
        )

    assert sampled_type_indices == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert env._chess_move_reset_state["count"] == 12
    assert env._chess_move_reset_state["balanced_success_count"] == 5


def test_generation_reset_uses_only_source_piece_types(monkeypatch):
    class Env:
        num_envs = 1
        device = "cpu"
        scene = {"rook_white": object(), "knight_white": object()}
        _chess_pickup_source_piece_types = frozenset({"rook"})
        _chess_move_reset_state = {
            "count": 1,
            "move": ("knight_white", (0, 0), (1, 2)),
        }

    env = Env()
    env_ids = torch.tensor([0], dtype=torch.long)
    captured = {}

    monkeypatch.setattr(chess_events, "move_piece_off_board", lambda *args, **kwargs: None)
    monkeypatch.setattr(chess_events, "place_piece_on_square", lambda *args, **kwargs: None)
    monkeypatch.setattr(chess_events, "_set_piece_awake", lambda *args, **kwargs: None)
    monkeypatch.setattr(chess_events, "color_board_squares", lambda *args, **kwargs: None)

    def sample_random(**kwargs):
        captured.update(kwargs)
        return "rook_white", (0, 0), (0, 1)

    monkeypatch.setattr(chess_events, "sample_chess_move_random", sample_random)

    move = chess_events.sample_chess_move_reset(
        env,
        env_ids,
        piece_names=["rook_white", "knight_white"],
        sampling_strategy="random",
        advance_on_success_only=False,
        min_pieces=1,
        max_pieces=1,
    )

    assert move == ("rook_white", (0, 0), (0, 1))
    assert captured["piece_names"] == ["rook_white"]
    assert env._chess_move_reset_state["count"] == 2
    # Unsupported geometry may still appear as a distractor; it is excluded
    # only from active pickup sampling.
    assert set(captured["available_piece_names"]) == {"rook_white", "knight_white"}
