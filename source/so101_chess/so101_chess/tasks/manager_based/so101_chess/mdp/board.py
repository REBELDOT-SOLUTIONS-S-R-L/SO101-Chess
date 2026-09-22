# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared chessboard geometry helpers.

The visual squares are deliberately not used as physics surfaces.  Their source
USD contains stale authored extents, and separate colliders on all 64 squares
create internal contact edges.  The task spawner creates one continuous surface
collider instead; these helpers make that collider the authoritative source of
the board height everywhere else in the task.
"""

from __future__ import annotations

from pxr import Gf, Usd, UsdGeom


BOARD_SURFACE_COLLIDER_NAME = "SurfaceCollider"


def board_surface_path(env_id: int) -> str:
    """Return the continuous board-collider path for one environment."""
    return f"/World/envs/env_{env_id}/Scene/ChessBoard/{BOARD_SURFACE_COLLIDER_NAME}"


def square_path(env_id: int, row: int, column: int) -> str:
    """Return a visual square's path for one environment."""
    return f"/World/envs/env_{env_id}/Scene/ChessBoard/Square_{row}_{column}"


def square_surface_position(
    stage: Usd.Stage,
    env_id: int,
    row: int,
    column: int,
    z_offset: float = 0.0,
) -> Gf.Vec3d:
    """Return a square center at the real continuous collision surface.

    XY comes from the square transform so recoloring/re-spacing stays aligned.
    Z comes exclusively from the continuous collider, whose canonical Cube
    extent is authored by the task spawner.
    """
    if not (0 <= row <= 7 and 0 <= column <= 7):
        raise ValueError(f"Invalid chess square: {(row, column)}")

    square_prim_path = square_path(env_id, row, column)
    square_prim = stage.GetPrimAtPath(square_prim_path)
    if not square_prim.IsValid():
        raise RuntimeError(f"Square prim does not exist: {square_prim_path}")

    surface_prim_path = board_surface_path(env_id)
    surface_prim = stage.GetPrimAtPath(surface_prim_path)
    if not surface_prim.IsValid():
        raise RuntimeError(f"Board surface collider does not exist: {surface_prim_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    square_center = xform_cache.GetLocalToWorldTransform(square_prim).Transform(Gf.Vec3d(0.0))

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    surface_bounds = bbox_cache.ComputeWorldBound(surface_prim).ComputeAlignedRange()
    surface_z = surface_bounds.GetMax()[2]
    return Gf.Vec3d(square_center[0], square_center[1], surface_z + z_offset)
