# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-time domain randomization for the SO-101 chess scene.

Camera poses are sampled around their configured pose on every reset, never
around their previous pose. This keeps the perturbation bounded over arbitrarily
long runs. Display-material randomization leaves contact and physics materials
untouched. One set of materials is created per environment to avoid shared USD
shader inputs making every clone change appearance at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# A single, low-contrast Isaac Sim library texture gives the white table a
# painted MDF-like grain. Its contrast is compressed in _preview_material.
_TABLE_TEXTURE = "textured_wall.png"
# USD Preview Surface fallback for the Isaac material-library plastic preset.
_PLASTIC_MDL = "OmniSurfacePresets.mdl"
_PLASTIC_PRESET = "OmniSurface_Plastic"
# Filament color supplied for the printed SO-101 robot. Keep the same hue in
# shader inputs so the rendered object remains readable under task lighting.
_ROBOT_FILAMENT_COLOR = (8 / 255, 10 / 255, 13 / 255)


def _texture_url() -> str:
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path(skip_check=True).rstrip("/")
    return f"{root}/Isaac/Samples/DR/Materials/Textures/{_TABLE_TEXTURE}"


_LIGHT_NAMES = ("TopLight", "FrontLight", "RightLight")
_PIECE_NAMES = (
    "pawn_white", "rook_white", "knight_white", "bishop_white", "queen_white", "king_white",
    "pawn_black", "rook_black", "knight_black", "bishop_black", "queen_black", "king_black",
)

_POSITION_AXES = ("x", "y", "z")
_ROTATION_AXES = ("roll", "pitch", "yaw")


def _validate_axis_ranges(
    ranges: dict[str, tuple[float, float]],
    allowed_axes: tuple[str, ...],
    label: str,
) -> None:
    unknown_axes = set(ranges) - set(allowed_axes)
    if unknown_axes:
        raise ValueError(f"{label} contains unsupported axes: {sorted(unknown_axes)}")
    for axis, bounds in ranges.items():
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError(f"{label}[{axis!r}] must be an increasing (min, max) pair")


def _resolve_env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()


def _sample_axis_offsets(
    count: int,
    ranges: dict[str, tuple[float, float]],
    axes: tuple[str, ...],
    device: str | torch.device,
) -> torch.Tensor:
    bounds = torch.tensor(
        [ranges.get(axis, (0.0, 0.0)) for axis in axes],
        dtype=torch.float32,
        device=device,
    )
    return math_utils.sample_uniform(
        bounds[:, 0],
        bounds[:, 1],
        (count, len(axes)),
        device=device,
    )


def randomize_fixed_camera_pose(
    env: ManagerBasedRLEnv,
    env_ids,
    camera_name: str = "top_camera",
    position_range: dict[str, tuple[float, float]] | None = None,
    rotation_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Randomize a fixed camera around its configured nominal pose.

    The configured ``CameraCfg.offset`` remains the immutable reference pose.
    Reset samples are added to that pose rather than the camera's current pose,
    preventing cumulative random-walk drift across episodes. Rotation offsets
    are applied in the camera's local frame.
    """
    position_range = position_range or {}
    rotation_range = rotation_range or {}
    _validate_axis_ranges(position_range, _POSITION_AXES, "position_range")
    _validate_axis_ranges(rotation_range, _ROTATION_AXES, "rotation_range")
    # Annotated-recording launchers may intentionally remove camera configs.
    if camera_name not in env.scene.keys():
        return

    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    camera = env.scene[camera_name]
    camera_env_ids = env_ids.to(device=camera.device)
    env_origins = env.scene.env_origins[camera_env_ids].to(device=camera.device)
    count = len(camera_env_ids)

    nominal_position = torch.tensor(
        camera.cfg.offset.pos,
        dtype=env_origins.dtype,
        device=camera.device,
    ).expand(count, -1)
    nominal_orientation = torch.tensor(
        camera.cfg.offset.rot,
        dtype=env_origins.dtype,
        device=camera.device,
    ).expand(count, -1)

    position_offsets = _sample_axis_offsets(
        count, position_range, _POSITION_AXES, camera.device
    ).to(dtype=env_origins.dtype)
    rotation_offsets = _sample_axis_offsets(
        count, rotation_range, _ROTATION_AXES, camera.device
    ).to(dtype=env_origins.dtype)

    positions = env_origins + nominal_position + position_offsets
    orientation_offsets = math_utils.quat_from_euler_xyz(
        rotation_offsets[:, 0],
        rotation_offsets[:, 1],
        rotation_offsets[:, 2],
    )
    orientations = math_utils.quat_mul(nominal_orientation, orientation_offsets)

    camera.set_world_poses(
        positions=positions,
        orientations=orientations,
        env_ids=camera_env_ids,
        convention=camera.cfg.offset.convention,
    )
    # Refresh the sensor pose buffers after changing the underlying prims.
    camera.reset(camera_env_ids)


def _uniform(low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand(()).item())


def _sample_enabled_lights(count: int, off_probability: float, min_active_lights: int) -> torch.Tensor:
    """Sample independent light availability while retaining a minimum count."""
    enabled = torch.rand(count) >= off_probability
    active_count = int(enabled.sum().item())
    if active_count < min_active_lights:
        disabled_indices = torch.where(~enabled)[0]
        selected = disabled_indices[torch.randperm(len(disabled_indices))[: min_active_lights - active_count]]
        enabled[selected] = True
    return enabled


def _enforce_minimum_height(
    position: tuple[float, float, float],
    minimum_height_m: float | None,
) -> tuple[float, float, float]:
    """Keep a sampled light above the configured scene-local height."""
    if minimum_height_m is None:
        return position
    return (position[0], position[1], max(position[2], minimum_height_m))


def _nominal_light_pose(env, path: str, prim: Usd.Prim):
    """Return and cache a light's authored local pose in ``(w, x, y, z)`` form."""
    pose_defaults = getattr(env, "_chess_light_pose_defaults", None)
    if pose_defaults is None:
        pose_defaults = {}
        env._chess_light_pose_defaults = pose_defaults
    if path not in pose_defaults:
        transform = Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation())
        translation = transform.GetTranslation()
        quaternion = transform.GetRotation().GetQuat()
        imaginary = quaternion.GetImaginary()
        pose_defaults[path] = (
            tuple(float(value) for value in translation),
            (float(quaternion.GetReal()), *(float(value) for value in imaginary)),
        )
    return pose_defaults[path]


def randomize_scene_lights(
    env: ManagerBasedRLEnv,
    env_ids,
    position_range: dict[str, tuple[float, float]] | None = None,
    rotation_range: dict[str, tuple[float, float]] | None = None,
    intensity_range: tuple[float, float] = (0.2, 2.0),
    temperature_range: tuple[float, float] = (2500.0, 9000.0),
    off_probability: float = 0.15,
    min_active_lights: int = 1,
    minimum_height_m: float | None = None,
) -> None:
    """Randomize every authored task light without removing any light prim.

    Position and rotation samples are offsets from the immutable authored local
    pose, so repeated resets cannot produce a random walk. Each light gets an
    independent intensity multiplier and can be disabled by setting its
    intensity to zero. At least ``min_active_lights`` remain illuminated.
    """
    position_range = position_range or {}
    rotation_range = rotation_range or {}
    _validate_axis_ranges(position_range, _POSITION_AXES, "position_range")
    _validate_axis_ranges(rotation_range, _ROTATION_AXES, "rotation_range")
    if intensity_range[0] <= 0 or intensity_range[0] > intensity_range[1]:
        raise ValueError("intensity_range must be positive and increasing")
    if temperature_range[0] <= 0 or temperature_range[0] > temperature_range[1]:
        raise ValueError("temperature_range must be positive and increasing")
    if not 0.0 <= off_probability <= 1.0:
        raise ValueError("off_probability must be between 0 and 1")
    if not 1 <= min_active_lights <= len(_LIGHT_NAMES):
        raise ValueError(f"min_active_lights must be between 1 and {len(_LIGHT_NAMES)}")

    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    stage = env.sim.stage
    light_defaults = getattr(env, "_chess_light_defaults", None)
    if light_defaults is None:
        light_defaults = {}
        env._chess_light_defaults = light_defaults

    for env_id in env_ids.cpu().tolist():
        scene_path = f"/World/envs/env_{env_id}/Scene"

        # The imported defaultGroundPlane contains a 100000-intensity sphere
        # light. Keep its prim but disable it so the task lights remain visible.
        ground_light = stage.GetPrimAtPath(f"{scene_path}/defaultGroundPlane/SphereLight")
        if ground_light.IsValid():
            UsdLux.LightAPI(ground_light).GetIntensityAttr().Set(0.0)

        light_count = len(_LIGHT_NAMES)
        position_offsets = _sample_axis_offsets(
            light_count, position_range, _POSITION_AXES, "cpu"
        )
        rotation_offsets = _sample_axis_offsets(
            light_count, rotation_range, _ROTATION_AXES, "cpu"
        )
        orientation_offsets = math_utils.quat_from_euler_xyz(
            rotation_offsets[:, 0],
            rotation_offsets[:, 1],
            rotation_offsets[:, 2],
        )
        enabled_lights = _sample_enabled_lights(
            light_count, off_probability, min_active_lights
        )
        temperature = _uniform(*temperature_range)
        sampled_intensities = {}
        sampled_multipliers = {}
        sampled_positions = {}
        sampled_orientations = {}

        for index, name in enumerate(_LIGHT_NAMES):
            path = f"{scene_path}/{name}"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"Chess scene light is missing: {path}")

            light = UsdLux.LightAPI(prim)
            intensity = light.GetIntensityAttr()
            if path not in light_defaults:
                light_defaults[path] = float(intensity.Get())
            multiplier = _uniform(*intensity_range)
            sampled_intensity = light_defaults[path] * multiplier if enabled_lights[index] else 0.0
            intensity.Set(sampled_intensity)
            light.GetEnableColorTemperatureAttr().Set(True)
            light.GetColorTemperatureAttr().Set(temperature)

            nominal_position, nominal_orientation = _nominal_light_pose(env, path, prim)
            sampled_position = tuple(
                nominal_position[axis] + float(position_offsets[index, axis].item())
                for axis in range(3)
            )
            sampled_position = _enforce_minimum_height(
                sampled_position, minimum_height_m
            )
            nominal_orientation_tensor = torch.tensor(
                nominal_orientation, dtype=orientation_offsets.dtype
            ).unsqueeze(0)
            sampled_orientation = math_utils.quat_mul(
                nominal_orientation_tensor, orientation_offsets[index].unsqueeze(0)
            )[0]
            sampled_orientation_tuple = tuple(float(value.item()) for value in sampled_orientation)
            sim_utils.standardize_xform_ops(
                prim,
                translation=sampled_position,
                orientation=sampled_orientation_tuple,
            )

            sampled_intensities[name] = sampled_intensity
            sampled_multipliers[name] = multiplier
            sampled_positions[name] = sampled_position
            sampled_orientations[name] = sampled_orientation_tuple

        if not hasattr(env, "chess_randomization_state"):
            env.chess_randomization_state = {}
        state = env.chess_randomization_state.setdefault(env_id, {})
        state.update(
            {
                "light_temperature_k": temperature,
                "light_intensities": sampled_intensities,
                "light_intensity_multipliers": sampled_multipliers,
                "light_positions": sampled_positions,
                "light_orientations_wxyz": sampled_orientations,
                "active_lights": [
                    name for index, name in enumerate(_LIGHT_NAMES) if enabled_lights[index]
                ],
            }
        )
        print(
            f"[CHESS DR] env={env_id} active_lights={state['active_lights']} "
            f"light_intensities={sampled_intensities} temperature_k={temperature:.0f}"
        )


def _preview_material(
    stage: Usd.Stage,
    path: str,
    color: tuple[float, float, float],
    roughness: float,
    texture_strength: float = 0.0,
):
    """Create a matte preview material, optionally with subtle library grain."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if texture_strength == 0.0:
        diffuse.DisconnectSource()
        diffuse.Set(Gf.Vec3f(*color))
    else:
        texture = UsdShade.Shader.Define(stage, f"{path}/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(_texture_url()))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        # The library image itself is mid-gray. Blend only a small amount of
        # its variation into the chosen white base so the table stays white.
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(texture_strength, texture_strength, texture_strength, 1.0)
        )
        texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(*(c - 0.7 * texture_strength for c in color), 0.0)
        )
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


def _plastic_material(stage: Usd.Stage, path: str, color: tuple[float, float, float], roughness: float):
    """Use Isaac Sim's plastic material preset with restrained printed finish."""
    material = _preview_material(stage, path, color, roughness)
    shader = UsdShade.Shader.Define(stage, f"{path}/PlasticShader")
    shader.SetSourceAsset(Sdf.AssetPath(_PLASTIC_MDL), "mdl")
    shader.SetSourceAssetSubIdentifier(_PLASTIC_PRESET, "mdl")
    shader.CreateInput("diffuse_reflection_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("subsurface_transmission_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("enable_diffuse_transmission", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("subsurface_weight", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specular_reflection_roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
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


def _bind_visible_geometry(stage: Usd.Stage, root_path: str, material, textured: bool = False) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Appearance target is missing: {root_path}")
    # A strong root binding also reaches referenced visual prims whose meshes
    # are not traversable yet while the scene is being initialized.
    UsdShade.MaterialBindingAPI(root).Bind(material, bindingStrength="strongerThanDescendants")
    found = False
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(prim).ComputeVisibility() != "invisible":
            if textured and prim.IsA(UsdGeom.Mesh):
                _ensure_mesh_uv(UsdGeom.Mesh(prim))
            elif textured and prim.IsA(UsdGeom.Cube):
                _ensure_cube_uv(prim)
            UsdShade.MaterialBindingAPI(prim).Bind(material, bindingStrength="strongerThanDescendants")
            found = True
    if not found:
        raise RuntimeError(f"No visible geometry found under {root_path}")


def _piece_root(env, name: str, env_id: int) -> str:
    """Use the actual spawned prim path, including the white pieces' short names."""
    return str(env.scene[name].root_physx_view.prim_paths[env_id])


def randomize_chess_appearance(
    env: ManagerBasedRLEnv,
    env_ids,
    light_intensity_range: tuple[float, float] = (0.2, 2.0),
    light_temperature_range: tuple[float, float] = (2500.0, 9000.0),
    light_position_range: dict[str, tuple[float, float]] | None = None,
    light_rotation_range: dict[str, tuple[float, float]] | None = None,
    light_off_probability: float = 0.15,
    min_active_lights: int = 1,
    light_minimum_height_m: float | None = None,
) -> None:
    """Vary lights, table, robot, pieces and board on each reset.

    Lights retain their authored prims and independently vary pose and
    intensity; a bounded subset may have zero intensity. The red source and
    green target markers and yellow board border keep their authored materials.
    Piece colors stay on their named side with one printed-plastic family.
    """
    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    randomize_scene_lights(
        env,
        env_ids,
        position_range=light_position_range,
        rotation_range=light_rotation_range,
        intensity_range=light_intensity_range,
        temperature_range=light_temperature_range,
        off_probability=light_off_probability,
        min_active_lights=min_active_lights,
        minimum_height_m=light_minimum_height_m,
    )

    stage = env.sim.stage

    for env_id in env_ids.cpu().tolist():
        scene_path = f"/World/envs/env_{env_id}/Scene"
        looks = f"{scene_path}/DomainRandomizationLooks"
        UsdGeom.Scope.Define(stage, looks)

        # Painted white MDF: one library texture family, with only slight
        # changes in shade, grain contrast and roughness between episodes.
        table_shade = _uniform(0.88, 0.97)
        table_grain = _uniform(0.025, 0.065)
        table = _preview_material(
            stage, f"{looks}/Table", (table_shade,) * 3,
            roughness=_uniform(0.65, 0.78), texture_strength=table_grain,
        )
        _bind_visible_geometry(stage, f"{scene_path}/TableBase", table, textured=True)

        # Match the filament hue with a matte plastic finish. Lift the shader
        # reflectance so its printed details remain visible under dim lighting.
        robot_scale = _uniform(2.0, 2.4)
        robot_color = tuple(channel * robot_scale for channel in _ROBOT_FILAMENT_COLOR)
        robot = _plastic_material(stage, f"{looks}/Robot", robot_color, roughness=_uniform(0.78, 0.85))
        _bind_visible_geometry(stage, f"{scene_path}/arm", robot)

        board = f"{scene_path}/ChessBoard"
        white = _uniform(0.78, 1.0)
        dark_gray = _uniform(0.12, 0.30)
        white_mat = _preview_material(stage, f"{looks}/WhiteSquares", (white,) * 3, roughness=0.5)
        dark_mat = _preview_material(stage, f"{looks}/DarkSquares", (dark_gray,) * 3, roughness=0.5)
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
            shade = _uniform(0.82, 0.97) if name.endswith("white") else _uniform(0.045, 0.095)
            color = (shade,) * 3
            material = _plastic_material(
                stage, f"{looks}/{name}", color, roughness=_uniform(0.55, 0.7)
            )
            _bind_visible_geometry(stage, _piece_root(env, name, env_id), material)
