# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Chess-piece type helpers shared by recording and Mimic selection."""

from __future__ import annotations

import re


PIECE_TYPES = ("pawn", "rook", "knight", "bishop", "queen", "king")
_COLOUR_TOKENS = {"white", "black"}


def normalize_piece_type(piece_name: str) -> str:
    """Return the geometry type for a colored chess-piece asset name.

    Color tokens may appear before or after the type, so names such as
    ``rook_white``, ``white_rook``, ``rook-black`` and ``black rook`` all
    normalize to ``rook``.
    """
    if not isinstance(piece_name, str) or not piece_name.strip():
        raise ValueError("piece_name must be a non-empty string")

    tokens = [token for token in re.split(r"[^a-z0-9]+", piece_name.lower()) if token]
    geometry_tokens = [token for token in tokens if token not in _COLOUR_TOKENS]
    matches = [piece_type for piece_type in PIECE_TYPES if piece_type in geometry_tokens]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot normalize chess piece type from {piece_name!r}; "
            f"expected exactly one of {PIECE_TYPES}."
        )
    return matches[0]
