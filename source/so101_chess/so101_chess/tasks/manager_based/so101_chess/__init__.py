# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-Chess-v0",
    entry_point=f"{__name__}.so101_chess_env:So101ChessEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_chess_cfg:So101ChessTaskCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-Chess-v0-Mimic",
    entry_point=f"{__name__}.so101_chess_env:So101ChessEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_chess_mimic_cfg:So101ChessMimicEnvCfg",
    },
)
