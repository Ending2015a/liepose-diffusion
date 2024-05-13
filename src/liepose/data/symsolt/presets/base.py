import imageio
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


class BasePreset:
  obj_index: int
  num_sym_poses: int
  sym_normals: np.ndarray
  sym_angles: np.ndarray
  sym_tangents: np.ndarray

  _sym_poses: np.ndarray = None

  @classmethod
  def is_ambiguous(cls, image):
    return True

  @classmethod
  def set_texture(cls, mesh, texture_path):
    img = np.asarray(imageio.imread(texture_path))
    mesh.textures = [o3d.geometry.Image(img)]
    uvs = np.asarray(mesh.triangle_uvs)
    uvs = np.clip(uvs, 0.001, 0.999)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
    mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(mesh.triangles))
    return mesh

  @classmethod
  def get_color_encoding(cls, verts):
    verts = np.array(verts)
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    colors = (verts - mn) / (mx - mn)
    return o3d.utility.Vector3dVector(colors)

  @classmethod
  def _symmetric_factory(
    cls,
    normals,
    tangents,
    angles,
  ):
    # normals: (n, 3)
    # tangents: (n, 3)
    # angles: (m,)
    assert normals.shape == tangents.shape
    n = len(normals)
    m = len(angles)

    # (n*m, 3)
    normals = np.stack([normals] * m, axis=1).reshape((-1, 3))
    tangents = np.stack([tangents] * m, axis=1).reshape((-1, 3))
    bitangents = np.cross(normals, tangents)
    angles = np.stack([angles] * n, axis=0).reshape((-1, 1))

    # normalize
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    tangents = tangents / np.linalg.norm(tangents, axis=-1, keepdims=True)
    bitangents = bitangents / np.linalg.norm(bitangents, axis=-1, keepdims=True)
    # construct coordinate frames
    u = np.array([normals[0], tangents[0], bitangents[0]], dtype=np.float64).T
    v = np.array([normals, tangents, bitangents], dtype=np.float64).transpose(1, 2, 0)

    r1 = Rotation.from_matrix(np.matmul(u, v.transpose(0, 2, 1)))
    r2 = Rotation.from_rotvec(angles * normals[0])
    r = r2 * r1
    return r.as_matrix()

  @classmethod
  def get_symmetric_poses(cls, pose_mat: np.ndarray) -> np.ndarray:
    """This function produces poses of the object by
    the given pose matrix

    Args:
      pose_mat (np.ndarray): pose matrix (*, 4, 4)

    Returns:
      np.ndarray: symmetric poses (*, n, 4, 4)
    """
    if cls._sym_poses is None:
      sym_rots = cls._symmetric_factory(
        normals=cls.sym_normals, angles=cls.sym_angles, tangents=cls.sym_tangents
      )
      poses = np.stack([np.eye(4)] * cls.num_sym_poses, axis=0)
      poses[..., :3, :3] = sym_rots
      cls._sym_poses = poses

    pose_mat = pose_mat[..., np.newaxis, :, :]
    return np.matmul(pose_mat, cls._sym_poses)

  @classmethod
  def create_mesh(cls):
    raise NotImplementedError

  @classmethod
  def create(
    cls, *args, color=None, centered=True, **kwargs
  ) -> o3d.geometry.TriangleMesh:
    """Create mesh

    Returns:
      o3d.geometry.TriangleMesh: mesh
    """
    mesh = cls.create_mesh(*args, **kwargs)
    mesh.compute_vertex_normals()

    if color is True:
      mesh.vertex_colors = cls.get_color_encoding(mesh.vertices)
    elif isinstance(color, (list, tuple, np.ndarray)):
      mesh = mesh.paint_uniform_color(color)

    if centered:
      mesh = mesh.translate((0, 0, 0), relative=False)
    return mesh
