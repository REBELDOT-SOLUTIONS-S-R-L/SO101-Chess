# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific event functions for so101_chess.

Add custom event/reset functions here. They will be available as `mdp.<func_name>`
in your task config's EventCfg.

Base events (reset_scene_to_default, etc.) are already available from isaaclab.envs.mdp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import omni.physx
import omni.usd
import torch
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdShade

from .board import square_surface_position

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def place_piece_on_square(
    env: ManagerBasedRLEnv,
    env_ids,
    piece_name: str,
    square: tuple[int, int] | list[int],
    z_offset: float = 0.001,
):
    """Place a rigid chess piece on a board square for the environments being reset."""
    row, column = square
    if not (0 <= row <= 7 and 0 <= column <= 7):
        raise ValueError(f"Invalid chess square: {square}")

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)

    if env_ids.numel() == 0:
        return

    target_positions = []
    for env_id in env_ids.cpu().tolist():
        position = square_surface_position(
            env.sim.stage,
            int(env_id),
            int(row),
            int(column),
            z_offset=z_offset,
        )
        target_positions.append(tuple(position))

    piece = env.scene[piece_name]
    target_position = torch.tensor(
        target_positions,
        dtype=piece.data.root_pos_w.dtype,
        device=env.device,
    )
    default_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        dtype=piece.data.root_quat_w.dtype,
        device=env.device,
    )
    orientation = default_quat.unsqueeze(0).repeat(len(target_positions), 1)
    target_pose = torch.cat((target_position, orientation), dim=-1)

    piece.write_root_pose_to_sim(target_pose, env_ids=env_ids)
    piece.write_root_velocity_to_sim(
        torch.zeros((len(target_positions), 6), dtype=target_pose.dtype, device=env.device),
        env_ids=env_ids,
    )


_DEFAULT_CHESS_PIECES = (
    "pawn_white",
    "rook_white",
    "knight_white",
    "bishop_white",
    "queen_white",
    "king_white",
    "pawn_black",
    "rook_black",
    "knight_black",
    "bishop_black",
    "queen_black",
    "king_black",
)


def _piece_name_lower(piece_name: str) -> str:
    return str(piece_name).lower()


def _piece_direction(piece_name: str) -> int:
    return 1 if "white" in _piece_name_lower(piece_name) else -1


def _in_bounds(square: tuple[int, int], board_size: int = 8) -> bool:
    row, col = square
    return 0 <= row < board_size and 0 <= col < board_size


def _add_unique_targets(targets: list[tuple[int, int]], candidate: tuple[int, int]) -> None:
    if _in_bounds(candidate) and candidate not in targets:
        targets.append(candidate)


def get_valid_chess_targets(
    piece_name: str,
    source_square: tuple[int, int],
    board_size: int = 8,
) -> list[tuple[int, int]]:
    """Return every valid destination square reachable from a piece on the source square.

    The function intentionally ignores occupancy and side-to-side constraints; it only
    enumerates the board positions that are reachable by the selected piece type.
    """
    if not isinstance(source_square, (tuple, list)) or len(source_square) != 2:
        raise ValueError(f"source_square must be a (row, col) tuple: {source_square}")

    row, col = int(source_square[0]), int(source_square[1])
    if not _in_bounds((row, col), board_size):
        raise ValueError(f"source_square is outside the board: {source_square}")

    name = _piece_name_lower(piece_name)
    targets: list[tuple[int, int]] = []

    if "pawn" in name:
        direction = _piece_direction(piece_name)
        forward_row = row + direction
        if _in_bounds((forward_row, col), board_size):
            targets.append((forward_row, col))
        for offset in (-1, 1):
            candidate = (row + direction, col + offset)
            if _in_bounds(candidate, board_size):
                targets.append(candidate)
    elif "knight" in name:
        for dr, dc in ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)):
            candidate = (row + dr, col + dc)
            if _in_bounds(candidate, board_size):
                targets.append(candidate)
    elif "bishop" in name:
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            r, c = row + dr, col + dc
            while _in_bounds((r, c), board_size):
                targets.append((r, c))
                r += dr
                c += dc
    elif "rook" in name:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = row + dr, col + dc
            while _in_bounds((r, c), board_size):
                targets.append((r, c))
                r += dr
                c += dc
    elif "queen" in name:
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = row + dr, col + dc
            while _in_bounds((r, c), board_size):
                targets.append((r, c))
                r += dr
                c += dc
    elif "king" in name:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                candidate = (row + dr, col + dc)
                if _in_bounds(candidate, board_size):
                    targets.append(candidate)
    else:
        raise ValueError(f"Unsupported chess piece: {piece_name}")

    return sorted({target for target in targets if target != (row, col)})


def sample_chess_move_sobol(
    seed: int | None = None,
    piece_names: list[str] | tuple[str, ...] | None = None,
    board_size: int = 8,
    available_piece_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Sample one legal chess move from a Sobol sequence.

    The piece, source square and destination square are generated as a single
    move tuple from the same Sobol draw, so they remain internally consistent.
    """
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())

    valid_moves = _enumerate_valid_chess_moves(piece_names, board_size, available_piece_names)
    engine = torch.quasirandom.SobolEngine(1, scramble=True, seed=int(seed))
    sample = engine.draw(1)[0].item()
    move_index = int(sample * len(valid_moves)) % len(valid_moves)
    return valid_moves[move_index]


def sample_chess_move_random(
    piece_names: list[str] | tuple[str, ...] | None = None,
    board_size: int = 8,
    available_piece_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Sample one legal chess move with the ordinary PyTorch RNG."""
    valid_moves = _enumerate_valid_chess_moves(piece_names, board_size, available_piece_names)
    move_index = int(torch.randint(len(valid_moves), (1,), dtype=torch.int64).item())
    return valid_moves[move_index]


def _enumerate_valid_chess_moves(
    piece_names: list[str] | tuple[str, ...] | None,
    board_size: int,
    available_piece_names: list[str] | tuple[str, ...] | None,
) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    """Enumerate the legal move population shared by all sampling strategies."""
    candidate_pieces = list(piece_names) if piece_names else list(_DEFAULT_CHESS_PIECES)
    if available_piece_names is not None:
        candidate_pieces = [name for name in candidate_pieces if name in available_piece_names]
    if not candidate_pieces:
        raise ValueError("piece_names must contain at least one chess piece name")

    valid_moves: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    for piece_name in candidate_pieces:
        for row in range(board_size):
            for col in range(board_size):
                source_square = (row, col)
                for target_square in get_valid_chess_targets(piece_name, source_square, board_size):
                    valid_moves.append((piece_name, source_square, target_square))

    if not valid_moves:
        raise ValueError("No valid chess moves were found for the requested piece set")
    return valid_moves


def move_piece_off_board(
    env: ManagerBasedRLEnv,
    env_ids,
    piece_name: str,
    offboard_pos: tuple[float, float, float] = (10.0, 10.0, 0.8),
) -> None:
    """Move a chess piece far away so it disappears from the board."""
    if not isinstance(piece_name, str):
        return
    scene_keys = set(env.scene.keys())
    if piece_name not in scene_keys:
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)

    if env_ids.numel() == 0:
        return

    piece = env.scene[piece_name]
    positions = piece.data.root_pos_w[env_ids].clone()
    default_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        dtype=piece.data.root_quat_w.dtype,
        device=env.device,
    )
    quaternions = default_quat.unsqueeze(0).repeat(len(env_ids), 1)
    positions[:, 0] = offboard_pos[0]
    positions[:, 1] = offboard_pos[1]
    positions[:, 2] = offboard_pos[2]
    piece.write_root_pose_to_sim(torch.cat((positions, quaternions), dim=-1), env_ids=env_ids)
    piece.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), dtype=positions.dtype, device=env.device),
        env_ids=env_ids,
    )

    # Parking a dynamic body out of view does not remove it from PhysX. Put it
    # to sleep so gravity does not keep five unused bodies in the solver for
    # the whole episode. Sleeping still permits immediate pose writes when a
    # later reset selects this piece again.
    _set_piece_awake(piece, env_ids, awake=False)


def _set_piece_awake(piece, env_ids: torch.Tensor, awake: bool) -> None:
    """Wake or sleep a rigid piece for the selected environments."""
    simulation = omni.physx.get_physx_simulation_interface()
    stage_id = omni.usd.get_context().get_stage_id()
    operation = simulation.wake_up if awake else simulation.put_to_sleep
    for env_id in env_ids.to(device="cpu", dtype=torch.long).tolist():
        prim_path = piece.root_physx_view.prim_paths[env_id]
        operation(stage_id, PhysicsSchemaTools.sdfPathToInt(prim_path))


def sample_chess_move_reset(
    env: ManagerBasedRLEnv,
    env_ids,
    piece_names: list[str] | tuple[str, ...] | None = None,
    seed: int = 0,
    board_size: int = 8,
    z_offset: float = 0.001,
    advance_on_success_only: bool = False,
    sampling_strategy: str = "sobol",
    min_pieces: int = 1,
    max_pieces: int = 6,
):
    """Sample a chess move, place the piece, and paint its board squares.

    When ``advance_on_success_only`` is true, repeated resets reuse the cached
    move until either the task's success termination fires or the recorder's
    successful-export counter increases. This lets an operator discard and
    retry a demonstration without consuming another move. ``sampling_strategy``
    accepts ``"sobol"`` for coverage-oriented recording or ``"random"`` for
    ordinary generation-time randomization.
    """
    if not 1 <= min_pieces <= max_pieces <= len(_DEFAULT_CHESS_PIECES):
        raise ValueError("piece count must satisfy 1 <= min_pieces <= max_pieces <= 12")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if env_ids.numel() == 0:
        return None

    scene_keys = set(env.scene.keys())
    available_piece_names = [name for name in _DEFAULT_CHESS_PIECES if name in scene_keys]
    if not available_piece_names and piece_names is not None:
        available_piece_names = [name for name in piece_names if isinstance(name, str) and name in scene_keys]

    if piece_names is None:
        piece_names = available_piece_names or list(_DEFAULT_CHESS_PIECES)

    state = getattr(env, "_chess_move_reset_state", None)
    if state is None:
        state = {"count": 0}
        setattr(env, "_chess_move_reset_state", state)

    episode_succeeded = False
    succeeded = getattr(env, "_chess_move_episode_succeeded", None)
    if succeeded is not None:
        if isinstance(succeeded, torch.Tensor):
            succeeded_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=succeeded.device)
            episode_succeeded = bool(torch.any(succeeded[succeeded_env_ids]).item())
            succeeded[succeeded_env_ids] = False
        else:
            episode_succeeded = bool(succeeded)
            setattr(env, "_chess_move_episode_succeeded", False)

    recorder_manager = getattr(env, "recorder_manager", None)
    successful_export_count = getattr(recorder_manager, "exported_successful_episode_count", None)
    if callable(successful_export_count):
        successful_export_count = successful_export_count()
    if successful_export_count is not None:
        if isinstance(successful_export_count, torch.Tensor):
            successful_export_count = int(successful_export_count.detach().cpu().item())
        else:
            successful_export_count = int(successful_export_count)
        previous_export_count = state.get("last_exported_successful_episode_count")
        episode_succeeded |= (
            previous_export_count is not None and successful_export_count > previous_export_count
        )
        state["last_exported_successful_episode_count"] = successful_export_count

    should_advance = (
        not advance_on_success_only
        or state.get("move") is None
        or episode_succeeded
    )
    if should_advance:
        if sampling_strategy == "sobol":
            move_seed = int(seed) + int(state["count"])
            state["move"] = sample_chess_move_sobol(
                seed=move_seed,
                piece_names=piece_names,
                board_size=board_size,
                available_piece_names=available_piece_names,
            )
        elif sampling_strategy == "random":
            state["move"] = sample_chess_move_random(
                piece_names=piece_names,
                board_size=board_size,
                available_piece_names=available_piece_names,
            )
        else:
            raise ValueError(
                f"Unsupported chess move sampling strategy: {sampling_strategy!r}. "
                "Expected 'sobol' or 'random'."
            )
        state["count"] += 1

    piece_name, source_square, target_square = state["move"]

    # Keep the goal and the grasp approach clear. For sliding pieces also
    # exclude squares along the move, so distractors cannot block the move.
    reserved = {tuple(source_square), tuple(target_square)}
    for center in (source_square, target_square):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                candidate = (center[0] + dr, center[1] + dc)
                if _in_bounds(candidate, board_size):
                    reserved.add(candidate)
    dr = target_square[0] - source_square[0]
    dc = target_square[1] - source_square[1]
    if "knight" not in piece_name and (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        step_r = (dr > 0) - (dr < 0)
        step_c = (dc > 0) - (dc < 0)
        for step in range(1, max(abs(dr), abs(dc))):
            reserved.add((source_square[0] + step * step_r, source_square[1] + step * step_c))

    available_distractors = [name for name in available_piece_names if name != piece_name]
    count = min(
        len(available_piece_names),
        int(torch.randint(min_pieces, max_pieces + 1, (1,)).item()),
    )
    shuffled_names = torch.randperm(len(available_distractors)).tolist()
    distractors = [available_distractors[index] for index in shuffled_names[:count - 1]]
    for other_piece in available_distractors:
        if other_piece not in distractors:
            move_piece_off_board(env, env_ids, other_piece)

    env.active_piece_name = piece_name
    env.active_source_square = tuple(source_square)
    env.active_target_square = tuple(target_square)
    if not hasattr(env, "chess_piece_squares"):
        env.chess_piece_squares = {}

    active_piece = env.scene[piece_name]
    place_piece_on_square(env, env_ids, piece_name, source_square, z_offset=z_offset)
    _set_piece_awake(active_piece, env_ids, awake=True)
    for env_id in env_ids.cpu().tolist():
        # Each environment gets independently shuffled, non-overlapping
        # distractor positions. Their names and number are shared because this
        # task's active-piece state is currently shared across environments.
        candidate_squares = [
            (row, col) for row in range(board_size) for col in range(board_size)
            if (row, col) not in reserved
        ]
        order = torch.randperm(len(candidate_squares)).tolist()
        placements = {piece_name: tuple(source_square)}
        for name, index in zip(distractors, order):
            square = candidate_squares[index]
            place_piece_on_square(
                env, torch.tensor([env_id], device=env.device), name, square, z_offset=z_offset
            )
            _set_piece_awake(env.scene[name], torch.tensor([env_id], device=env.device), awake=True)
            placements[name] = square
        env.chess_piece_squares[env_id] = placements

    color_board_squares(
        env,
        env_ids,
        reset_to_default=True,
        red_squares=[source_square],
        green_squares=[target_square],
    )
    print(
        f"[CHESS RESET] sampler={sampling_strategy} "
        f"sample={'advanced' if should_advance else 'reused'} "
        f"selected_piece={piece_name} source={source_square} target={target_square} "
        f"piece_count={count}"
    )
    return piece_name, source_square, target_square


# Fallback diffuse colors, used only when the scene has no authored red/green
# material to reuse (then we create a UsdPreviewSurface one on the fly).
_RED_FALLBACK = (0.80, 0.02, 0.02)
_GREEN_FALLBACK = (0.02, 0.55, 0.05)

# Square prim path -> the material it was originally bound to (its white/black
# `visual_material_NN`). Captured the first time each square is seen, BEFORE we
# override anything, so a reset can restore the board's true look instead of a
# flat repaint. Persists across resets for the lifetime of the process.
_ORIGINAL_SQUARE_MATERIAL: dict[str, str] = {}


def _direct_binding_path(prim):
    """Return the Sdf path of `prim`'s direct material binding, or None."""
    if not prim.IsValid():
        return None
    rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
    targets = rel.GetTargets() if rel else []
    return targets[0] if targets else None


def _ensure_material(stage, path: str, color: tuple[float, float, float]) -> UsdShade.Material:
    """Return the material at `path`, creating a UsdPreviewSurface one if missing."""
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        return UsdShade.Material(prim)

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _resolve_material(stage, looks_path, name_hint, canonical_name, fallback_color):
    """Find a material under `looks_path` whose name contains `name_hint`
    (case-insensitive), reusing the scene's authored one; otherwise create it."""
    looks = stage.GetPrimAtPath(looks_path)
    if looks.IsValid():
        for child in looks.GetChildren():
            if child.IsA(UsdShade.Material) and name_hint in child.GetName().lower():
                return UsdShade.Material(child)
    return _ensure_material(stage, f"{looks_path}/{canonical_name}", fallback_color)


def _bind_material(prim, material) -> bool:
    """Bind `material` directly on `prim` (the Square is a Cube gprim, not a Mesh)."""
    if not prim.IsValid() or material is None or not material.GetPrim().IsValid():
        return False
    UsdShade.MaterialBindingAPI(prim).Bind(
        material, bindingStrength="strongerThanDescendants"
    )
    return True


def color_board_squares(
    env: ManagerBasedRLEnv,
    env_ids,
    reset_to_default: bool = True,
    red_squares: list[tuple[int, int]] | None = None,
    green_squares: list[tuple[int, int]] | None = None,
):
    """Recolor the board on reset: restore every square to its original material,
    then paint the red (piece) square and green (target) square.

    The Square prims are ``Cube`` gprims (not Meshes) living under the per-env
    path ``/World/envs/env_<id>/Scene/ChessBoard``; the material is bound
    directly on the Cube.
    """

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)

    if env_ids.numel() == 0:
        return

    stage = env.sim.stage
    red_squares = red_squares or []
    green_squares = green_squares or []

    for env_id in env_ids.cpu().tolist():
        board = f"/World/envs/env_{env_id}/Scene/ChessBoard"
        looks = f"{board}/Looks"

        red_mat = _resolve_material(stage, looks, "red", "RedMat", _RED_FALLBACK)
        green_mat = _resolve_material(stage, looks, "green", "GreenMat", _GREEN_FALLBACK)

        # Capture each square's original material once, and restore it so no
        # red/green from a previous episode lingers (exactly 1 red + 1 green).
        for i in range(8):
            for j in range(8):
                sq_path = f"{board}/Square_{i}_{j}"
                sq = stage.GetPrimAtPath(sq_path)
                if not sq.IsValid():
                    continue
                if sq_path not in _ORIGINAL_SQUARE_MATERIAL:
                    orig = _direct_binding_path(sq)
                    if orig is not None:
                        _ORIGINAL_SQUARE_MATERIAL[sq_path] = str(orig)
                if reset_to_default:
                    orig = _ORIGINAL_SQUARE_MATERIAL.get(sq_path)
                    if orig is not None:
                        _bind_material(sq, UsdShade.Material(stage.GetPrimAtPath(orig)))

        for row, col in red_squares:
            _bind_material(stage.GetPrimAtPath(f"{board}/Square_{row}_{col}"), red_mat)
        for row, col in green_squares:
            _bind_material(stage.GetPrimAtPath(f"{board}/Square_{row}_{col}"), green_mat)
