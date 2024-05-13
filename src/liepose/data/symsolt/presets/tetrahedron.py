import numpy as np
import open3d as o3d

from .base import BasePreset

obj_index = 0

sym_normals: np.ndarray = np.array(
  [
    [-0.0, 0.0, -1.0],
    [0.47140452, -0.81649658, 0.33333333],
    [0.47140452, 0.81649658, 0.33333333],
    [-0.94280904, 0.0, 0.33333333],
  ],
  dtype=np.float64,
)
sym_tangents: np.ndarray = np.array(
  [
    [1.0, 0.0, 0.0],
    [0.83333333, 0.28867513, -0.47140452],
    [0.83333333, -0.28867513, -0.47140452],
    [-0.16666667, 0.8660254, -0.47140452],
  ],
  dtype=np.float64,
)

sym_angles = np.radians(np.array([-120, 0, 120], dtype=np.float64))

num_sym_poses = len(sym_angles) * len(sym_normals)


class TetrahedronPreset(BasePreset):
  obj_index: int = obj_index
  num_sym_poses: int = num_sym_poses

  sym_normals: np.ndarray = sym_normals
  sym_tangents: np.ndarray = sym_tangents
  sym_angles: np.ndarray = sym_angles

  @classmethod
  def create_mesh(cls, size=1.0):
    mesh = o3d.geometry.TriangleMesh.create_tetrahedron(
      radius=size / 1.3,
      create_uv_map=True,
    )
    return mesh
