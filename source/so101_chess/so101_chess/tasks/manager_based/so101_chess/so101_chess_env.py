# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task environment that installs the chess dataset recorder contract."""

from so101_chess.base_il_env.base_il_env import BaseILEnv

from .recorders import configure_chess_recorders


class So101ChessEnv(BaseILEnv):
    """Base IL environment with task-local recorder customizations."""

    def __init__(self, cfg, *args, **kwargs):
        configure_chess_recorders(cfg)
        super().__init__(cfg, *args, **kwargs)
