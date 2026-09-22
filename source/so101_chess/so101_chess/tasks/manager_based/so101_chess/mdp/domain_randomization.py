# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-time visual randomization for the SO-101 chess scene.

Only display materials are rebound. Contact and physics materials are left
untouched. One set of materials is created per environment to avoid shared USD
shader inputs making every clone change appearance at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Textures shipped in the Isaac Sim asset library. The asset root can be a
# local asset pack or a Nucleus URL, depending on the user's Isaac setup.
_LIBRARY_TEXTURES = {
    "marble": "marble_tile.png",
    "wall": "textured_wall.png",
}
_FINISHES = ("smooth", "marble", "wall")


def _texture_url(finish: str) -> str:
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path(skip_check=True).rstrip("/")
    return f"{root}/Isaac/Samples/DR/Materials/Textures/{_LIBRARY_TEXTURES[finish]}"


_LIGHT_NAMES = ("TopLight", "FrontLight", "RightLight")
_PIECE_NAMES = (
    "pawn_white", "rook_white", "knight_white", "bishop_white", "queen_white", "king_white",
    "pawn_black", "rook_black", "knight_black", "bishop_black", "queen_black", "king_black",
)


def _uniform(low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand(()).item())


def _preview_material(stage: Usd.Stage, path: str, color: tuple[float, float, float], finish: str):
    """Create or update a USD Preview Surface with a plain or mapped finish."""
    if finish not in _FINISHES:
        raise ValueError(f"Unknown visual finish: {finish}")
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        {"smooth": 0.5, "marble": 0.45, "wall": 0.75}[finish]
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if finish == "smooth":
        diffuse.DisconnectSource()
        diffuse.Set(Gf.Vec3f(*color))
    else:
        texture = UsdShade.Shader.Define(stage, f"{path}/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(_texture_url(finish)))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*color, 1.0))
        uv = UsdShade.Shader.Define(stage, f"{path}/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        diffuse.ConnectToSource(texture.ConnectableAPI(), "rgb")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _ensure_mesh_uv(mesh: UsdGeom.Mesh) -> None:
    """Add planar UVs when imported meshes have no texture coordinates."""
    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    if primvars.GetPrimvar("st").IsDefined():
        return
    points = mesh.GetPointsAttr().Get()
    if not points:
        return
    spans = [max(float(p[i]) for p in points) - min(float(p[i]) for p in points) for i in range(3)]
    axes = sorted(range(3), key=lambda axis: spans[axis], reverse=True)[:2]
    minima = [min(float(p[axis]) for p in points) for axis in axes]
    uv = [
        Gf.Vec2f(*((float(p[axis]) - minima[k]) / max(spans[axis], 1e-9) for k, axis in enumerate(axes)))
        for p in points
    ]
    primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(uv)


def _ensure_cube_uv(prim) -> None:
    """Give authored USD cubes one UV tile on each face."""
    primvars = UsdGeom.PrimvarsAPI(prim)
    if primvars.GetPrimvar("st").IsDefined():
        return
    face = (Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1))
    primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        list(face) * 6
    )


def _bind_visible_geometry(stage: Usd.Stage, root_path: str, material, finish: str) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Appearance target is missing: {root_path}")
    # A strong root binding also reaches referenced visual prims whose meshes
    # are not traversable yet while the scene is being initialized.
    UsdShade.MaterialBindingAPI(root).Bind(material, bindingStrength="strongerThanDescendants")
    found = False
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(prim).ComputeVisibility() != "invisible":
            if finish != "smooth" and prim.IsA(UsdGeom.Mesh):
                _ensure_mesh_uv(UsdGeom.Mesh(prim))
            elif finish != "smooth" and prim.IsA(UsdGeom.Cube):
                _ensure_cube_uv(prim)
            UsdShade.MaterialBindingAPI(prim).Bind(material, bindingStrength="strongerThanDescendants")
            found = True
    if not found:
        raise RuntimeError(f"No visible geometry found under {root_path}")


def _piece_root(env, name: str, env_id: int) -> str:
    """Use the actual spawned prim path, including the white pieces' short names."""
    return str(env.scene[name].root_physx_view.prim_paths[env_id])


def _random_finish() -> str:
    return _FINISHES[int(torch.randint(len(_FINISHES), (1,)).item())]


def randomize_chess_appearance(
    env: ManagerBasedRLEnv,
    env_ids,
    light_intensity_range: tuple[float, float] = (0.2, 2.0),
    light_temperature_range: tuple[float, float] = (2500.0, 9000.0),
) -> None:
    """Vary lights, table, robot, pieces and board on each reset.

    All three lights share one sampled temperature and intensity multiplier
    so the effect remains visible in a camera image. The red source and green
    target markers and yellow board border keep their authored materials.
    Piece colors stay on their named side while varying shade and finish.
    """
    if light_intensity_range[0] <= 0 or light_intensity_range[0] > light_intensity_range[1]:
        raise ValueError("light_intensity_range must be positive and increasing")
    if light_temperature_range[0] <= 0 or light_temperature_range[0] > light_temperature_range[1]:
        raise ValueError("light_temperature_range must be positive and increasing")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device)
    if env_ids.numel() == 0:
        return

    stage = env.sim.stage
    light_defaults = getattr(env, "_chess_light_defaults", None)
    if light_defaults is None:
        light_defaults = {}
        env._chess_light_defaults = light_defaults

    for env_id in env_ids.cpu().tolist():
        scene_path = f"/World/envs/env_{env_id}/Scene"
        looks = f"{scene_path}/DomainRandomizationLooks"
        UsdGeom.Scope.Define(stage, looks)

        # The imported defaultGroundPlane contains a 100000-intensity sphere
        # light. It overwhelms the three authored task lights, making changes
        # to their intensity and temperature nearly invisible in camera images.
        ground_light = stage.GetPrimAtPath(f"{scene_path}/defaultGroundPlane/SphereLight")
        if ground_light.IsValid():
            UsdLux.LightAPI(ground_light).GetIntensityAttr().Set(0.0)

        brightness = _uniform(*light_intensity_range)
        temperature = _uniform(*light_temperature_range)
        sampled_lights = {}
        for name in _LIGHT_NAMES:
            path = f"{scene_path}/{name}"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"Chess scene light is missing: {path}")
            light = UsdLux.LightAPI(prim)
            intensity = light.GetIntensityAttr()
            if path not in light_defaults:
                light_defaults[path] = float(intensity.Get())
            sampled_intensity = light_defaults[path] * brightness
            intensity.Set(sampled_intensity)
            light.GetEnableColorTemperatureAttr().Set(True)
            light.GetColorTemperatureAttr().Set(temperature)
            sampled_lights[name] = sampled_intensity
        if not hasattr(env, "chess_randomization_state"):
            env.chess_randomization_state = {}
        env.chess_randomization_state[env_id] = {
            "light_temperature_k": temperature,
            "light_intensities": sampled_lights,
            "light_brightness_multiplier": brightness,
        }
        print(
            f"[CHESS DR] env={env_id} light_intensities={sampled_lights} "
            f"temperature_k={temperature:.0f}"
        )

        table_shade = _uniform(0.78, 1.0)
        table_finish = _random_finish()
        table = _preview_material(stage, f"{looks}/Table", (table_shade,) * 3, table_finish)
        _bind_visible_geometry(stage, f"{scene_path}/TableBase", table, table_finish)

        robot_shade = _uniform(0.025, 0.13)
        robot_color = (robot_shade,) * 3
        robot_finish = _random_finish()
        robot = _preview_material(stage, f"{looks}/Robot", robot_color, robot_finish)
        _bind_visible_geometry(stage, f"{scene_path}/arm", robot, robot_finish)

        board = f"{scene_path}/ChessBoard"
        white = _uniform(0.78, 1.0)
        dark_gray = _uniform(0.12, 0.30)
        white_mat = _preview_material(stage, f"{looks}/WhiteSquares", (white,) * 3, "smooth")
        dark_mat = _preview_material(stage, f"{looks}/DarkSquares", (dark_gray,) * 3, "smooth")
        markers = {tuple(getattr(env, "active_source_square", (-1, -1))),
                   tuple(getattr(env, "active_target_square", (-1, -1)))}
        for row in range(8):
            for col in range(8):
                if (row, col) not in markers:
                    square = stage.GetPrimAtPath(f"{board}/Square_{row}_{col}")
                    UsdShade.MaterialBindingAPI(square).Bind(
                        white_mat if (row + col) % 2 == 0 else dark_mat
                    )

        for name in _PIECE_NAMES:
            if name not in env.scene.keys():
                continue
            shade = _uniform(0.73, 1.0) if name.endswith("white") else _uniform(0.035, 0.16)
            color = (shade,) * 3
            finish = _random_finish()
            material = _preview_material(stage, f"{looks}/{name}", color, finish)
            _bind_visible_geometry(stage, _piece_root(env, name, env_id), material, finish)
