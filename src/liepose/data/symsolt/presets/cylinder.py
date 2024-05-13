import numpy as np
import open3d as o3d

from .base import BasePreset

obj_index = 4
num_sym_poses = 720

sym_normals: np.ndarray = np.array([[0, 0, 1], [0, 0, -1]], dtype=np.float64)

sym_tangents: np.ndarray = np.array(
  [
    [-1, 0, 0],
    [1, 0, 0],
  ]
)

sym_angles = np.radians(np.linspace(180, -180, num_sym_poses // 2, endpoint=False))


class CylinderPreset(BasePreset):
  obj_index: int = obj_index
  num_sym_poses: int = num_sym_poses

  sym_normals: np.ndarray = sym_normals
  sym_tangents: np.ndarray = sym_tangents
  sym_angles: np.ndarray = sym_angles

  @classmethod
  def create_mesh(cls, size=1.0):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
      radius=size / 2,
      height=size,
      resolution=360,
      create_uv_map=True,
    )
    return mesh
