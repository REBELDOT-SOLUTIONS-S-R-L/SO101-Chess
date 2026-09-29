#!/usr/bin/env python3
"""Build centered, dimension-matched USD assets from the low-poly chess STLs.

Run with the Isaac Sim Python environment::

    /home/roboticslab/IsaacTools/.venv/bin/python \
        scripts/convert_lowpoly_chess_assets.py

The legacy USDs are used only as dimension references. They are not modified.
The generated meshes retain the legacy assets' authored dimensions so the
task's existing 0.0008 spawn scale remains unchanged.
"""

from __future__ import annotations

import argparse
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from isaacsim import SimulationApp


@dataclass(frozen=True)
class PieceSpec:
    name: str
    stl_name: str
    reference_usd_name: str
    yaw_degrees: float = 0.0

    @property
    def output_usd_name(self) -> str:
        return f"{self.name}_lowpoly.usd"


PIECES = (
    PieceSpec("Pawn", "1 Pawn.STL", "Pawn_collision.usd"),
    PieceSpec("Rook", "2 Rook.STL", "Rook_collision.usd"),
    # The STL's asymmetric upper body starts near +115 degrees. The task needs
    # it at +180 degrees relative to the asset frame, so bake in a +65 degree
    # correction before dimension matching. Rotating first also prevents the
    # non-uniform XY scale from skewing the requested heading.
    PieceSpec("Knight", "3 Knight.STL", "Knight_collision.usd", yaw_degrees=65.0),
    PieceSpec("Bishop", "4 Bishop.STL", "Bishop_collision.usd"),
    PieceSpec("Queen", "5 Queen.STL", "Queen_collision.usd"),
    PieceSpec("King", "6 King.STL", "King_collision.usd"),
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECTS_DIR = REPO_ROOT / "source/so101_chess/so101_chess/assets/objects"
LOWPOLY_DIR = OBJECTS_DIR / "lowpoly_chess_pieces"
REFERENCE_DIR = OBJECTS_DIR / "chesspieces"
SDF_RESOLUTION = 128


def _load_stl(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=True)
    geometries = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
    if not geometries:
        raise ValueError(f"No mesh geometry found in {path}")

    mesh = trimesh.util.concatenate(geometries)
    if not mesh.is_watertight:
        raise ValueError(f"SDF collision requires a watertight mesh: {path}")
    if not mesh.is_winding_consistent:
        raise ValueError(f"Mesh winding is inconsistent: {path}")

    # The source STLs are consistently wound inward. Positive volume gives
    # both rendering and SDF cooking outward-facing triangle winding.
    if mesh.volume < 0.0:
        mesh.invert()
    return mesh


def _support_origin(vertices: np.ndarray) -> np.ndarray:
    """Return the center of the bottom support ring, with Z on its base."""
    bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
    height = bounds[1, 2] - bounds[0, 2]
    tolerance = max(height * 1.0e-4, 1.0e-6)
    support = vertices[vertices[:, 2] <= bounds[0, 2] + tolerance]
    if len(support) < 3:
        raise ValueError("Could not identify at least three vertices on the support ring")
    support_bounds = np.vstack((support.min(axis=0), support.max(axis=0)))
    return np.array(
        [
            0.5 * (support_bounds[0, 0] + support_bounds[1, 0]),
            0.5 * (support_bounds[0, 1] + support_bounds[1, 1]),
            bounds[0, 2],
        ],
        dtype=np.float64,
    )


def _rotate_about_z(vertices: np.ndarray, yaw_degrees: float) -> np.ndarray:
    """Rotate vertices around their source origin without changing winding."""
    if yaw_degrees == 0.0:
        return vertices.copy()
    angle = math.radians(yaw_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return vertices @ rotation.T


def _reference_extents(path: Path, Usd, UsdGeom) -> np.ndarray:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Could not open reference USD: {path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    bounds = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    return np.asarray(tuple(bounds.GetSize()), dtype=np.float64)


def _write_usd(
    output_path: Path,
    piece: PieceSpec,
    vertices: np.ndarray,
    faces: np.ndarray,
    source_extents: np.ndarray,
    target_extents: np.ndarray,
    scale_factors: np.ndarray,
    Gf,
    Kind,
    PhysxSchema,
    Usd,
    UsdGeom,
    UsdPhysics,
    Vt,
) -> None:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.usd")
    if temporary_path.exists():
        temporary_path.unlink()

    stage = Usd.Stage.CreateNew(str(temporary_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)

    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root.GetPrim()).SetKind(Kind.Tokens.component)
    root.GetPrim().SetAssetInfoByKey("name", f"{piece.name} low-poly chess piece")
    root.GetPrim().SetCustomDataByKey("sourceStl", piece.stl_name)
    root.GetPrim().SetCustomDataByKey("dimensionReference", piece.reference_usd_name)
    root.GetPrim().SetCustomDataByKey("sourceExtents", Gf.Vec3d(*source_extents))
    root.GetPrim().SetCustomDataByKey("targetExtents", Gf.Vec3d(*target_extents))
    root.GetPrim().SetCustomDataByKey("scaleFactors", Gf.Vec3d(*scale_factors))
    root.GetPrim().SetCustomDataByKey("sourceYawDegrees", float(piece.yaw_degrees))

    UsdGeom.Xform.Define(stage, f"/Root/{piece.name}")
    usd_mesh = UsdGeom.Mesh.Define(stage, f"/Root/{piece.name}/mesh")
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).reshape(-1)))
    usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    usd_mesh.CreateDoubleSidedAttr(False)

    mesh_bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0))).astype(np.float32)
    usd_mesh.CreateExtentAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(*(float(value) for value in mesh_bounds[0])),
                Gf.Vec3f(*(float(value) for value in mesh_bounds[1])),
            ]
        )
    )
    usd_mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.65, 0.65, 0.65)]))

    # Flat, per-face normals retain the intended faceted low-poly appearance.
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    face_varying_normals = np.repeat(normals.astype(np.float32), 3, axis=0)
    usd_mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(face_varying_normals))
    usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

    collision_prim = usd_mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(collision_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
    mesh_collision.CreateApproximationAttr().Set("sdf")
    sdf_collision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(collision_prim)
    sdf_collision.CreateSdfResolutionAttr().Set(SDF_RESOLUTION)
    sdf_collision.CreateSdfSubgridResolutionAttr().Set(6)
    sdf_collision.CreateSdfBitsPerSubgridPixelAttr().Set("BitsPerPixel16")

    stage.GetRootLayer().Save()
    os.replace(temporary_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--piece",
        action="append",
        default=[],
        metavar="NAME",
        help="Convert only this piece (case-insensitive). May be repeated; the default converts all pieces.",
    )
    args = parser.parse_args()
    requested_names = {name.strip().lower() for name in args.piece}
    known_names = {piece.name.lower() for piece in PIECES}
    unknown_names = requested_names - known_names
    if unknown_names:
        parser.error(f"unknown piece name(s): {', '.join(sorted(unknown_names))}")

    app = SimulationApp({"headless": True})
    try:
        # USD and PhysX schemas become available after SimulationApp starts.
        from pxr import Gf, Kind, PhysxSchema, Usd, UsdGeom, UsdPhysics, Vt

        for piece in PIECES:
            if requested_names and piece.name.lower() not in requested_names:
                continue
            source_path = LOWPOLY_DIR / piece.stl_name
            reference_path = REFERENCE_DIR / piece.reference_usd_name
            output_path = LOWPOLY_DIR / piece.output_usd_name

            mesh = _load_stl(source_path)
            source_vertices = _rotate_about_z(
                np.asarray(mesh.vertices, dtype=np.float64), piece.yaw_degrees
            )
            source_extents = np.ptp(source_vertices, axis=0)
            target_extents = _reference_extents(reference_path, Usd, UsdGeom)
            scale_factors = target_extents / source_extents
            origin = _support_origin(source_vertices)
            vertices = (source_vertices - origin) * scale_factors

            _write_usd(
                output_path,
                piece,
                vertices,
                np.asarray(mesh.faces, dtype=np.int64),
                source_extents,
                target_extents,
                scale_factors,
                Gf,
                Kind,
                PhysxSchema,
                Usd,
                UsdGeom,
                UsdPhysics,
                Vt,
            )
            summary = (
                f"{piece.name:6s}: {len(mesh.faces):3d} triangles, "
                f"target extents {target_extents.tolist()}, "
                f"scale {scale_factors.tolist()} -> {output_path.name}\n"
            )
            # Kit redirects sys.stdout; writing to its original descriptor
            # keeps progress visible in both terminals and build logs.
            os.write(1, summary.encode())
    except Exception:
        # Kit redirects Python's standard streams, so use the original file
        # descriptor to keep conversion failures visible in headless runs.
        os.write(2, traceback.format_exc().encode())
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
