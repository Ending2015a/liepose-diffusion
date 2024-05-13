import imageio
import numpy as np
import open3d as o3d

from .base import BasePreset

obj_index = 1

sym_normals = np.array(
  [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
  dtype=np.float64,
)

sym_tangents = np.array(
  [
    [0, -1, 0],
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
  ],
  dtype=np.float64,
)

sym_angles = np.array([-np.pi / 2, 0, np.pi / 2, np.pi], dtype=np.float64)

num_sym_poses = len(sym_angles) * len(sym_normals)


class CubePreset(BasePreset):
  obj_index: int = obj_index
  num_sym_poses: int = num_sym_poses
  sym_normals: np.ndarray = sym_normals
  sym_tangents: np.ndarray = sym_tangents
  sym_angles: np.ndarray = sym_angles

  @classmethod
  def set_texture(cls, cube, texture_path, map_type="cube"):
    img = np.asarray(imageio.imread(texture_path))
    cube.textures = [o3d.geometry.Image(img)]
    cube.triangle_material_ids = o3d.utility.IntVector([0] * len(cube.triangles))

    def get_uv(uv, xoff, yoff, rot=0, hflip=False):
      uv_y = 0.333
      uv_x = 0.25
      for i in range(rot):
        uv = np.stack([uv[:, 1], uv[:, 0]], axis=-1)
        uv[:, 0] = 0.5 - (uv[:, 0] - 0.5)
      if hflip:
        uv[:, 0] = 0.5 - (uv[:, 0] - 0.5)
      return (uv + np.array([xoff, yoff])) * np.array([uv_x, uv_y])

    uvs = np.asarray(cube.triangle_uvs)
    if map_type == "cube":
      uvs[0:6] = get_uv(uvs[0:6], 1, 2)  # down
      uvs[6:12] = get_uv(uvs[6:12], 0, 1, rot=1)  # left
      uvs[12:18] = get_uv(uvs[12:18], 1, 0, rot=2, hflip=True)  # up
      uvs[18:24] = get_uv(uvs[18:24], 2, 1, rot=1, hflip=True)  # right
      uvs[24:30] = get_uv(uvs[24:30], 3, 1, hflip=True)  # forward (+z)
      uvs[30:36] = get_uv(uvs[30:36], 1, 1)  # backward (-z)
    cube.triangle_uvs = o3d.utility.Vector2dVector(uvs)
    return cube

  @classmethod
  def create_mesh(cls, size=1.0):
    mesh = o3d.geometry.TriangleMesh.create_box(
      width=size,
      height=size,
      depth=size,
      create_uv_map=True,
      map_texture_to_each_face=True,
    )
    return mesh
