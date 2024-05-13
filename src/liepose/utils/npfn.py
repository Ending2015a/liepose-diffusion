import numpy as np
from scipy.spatial.transform import Rotation


def homogeneous(p):
  """Convert pixel to homogeneous coordinate"""
  if p.shape[-1] == 2:
    return np.concatenate([p, np.ones_like(p[..., :1])], axis=-1)
  if p.shape[-1] == 3:
    return p / p[..., 2:3]
  raise ValueError(f"p.shape[-1] should be either 2 or 3, p.shape: {p.shape}")


def normalize(vec, eps=1e-6):
  return vec / np.linalg.norm(vec + eps, axis=-1, keepdims=True)


def get_obj_ray(cam: np.ndarray, pixel: np.ndarray):
  """Project pixels into 3D ray in homogeneous space"""
  point = homogeneous(pixel)
  obj_ray = homogeneous(point @ np.linalg.inv(cam).T)
  return obj_ray


def get_allo_rotation(
  obj_ray: np.ndarray, cam_ray: np.ndarray = (0, 0, 1), eps: float = 1e-6
):
  """Calculate allocentric rotations from the given target translation

  all = R * ego

  Args:
    tran (np.ndarray): _description_
    cam (np.array, optional): _description_. Defaults to np.float32).

  Return:
    np.ndarray: allocentric rotations in quaternion format (wxyz)
  """
  cam_ray = np.asarray(cam_ray, dtype=np.float32)
  cam_ray = normalize(cam_ray, eps=eps)
  # (..., 3)
  obj_ray = np.asarray(obj_ray, dtype=np.float32)
  obj_ray = normalize(obj_ray, eps=eps)

  ang = np.arccos(np.clip((cam_ray * obj_ray).sum(axis=-1), -1, 1))  # (...,)
  axis = np.cross(cam_ray, obj_ray)  # (..., 3)
  # avoid zero norm axis
  axis = normalize(axis, eps=eps)

  sin_ang = np.sin(ang / 2.0)
  cos_ang = np.cos(ang / 2.0)

  quat = np.stack(
    [
      cos_ang,
      -axis[..., 0] * sin_ang,
      -axis[..., 1] * sin_ang,
      -axis[..., 2] * sin_ang,
    ],
    axis=-1,
  )

  return quat


def wxyz_to_xyzw(quat):
  return np.roll(quat, -1, axis=-1)


def xyzw_to_wxyz(quat):
  return np.roll(quat, 1, axis=-1)


def wxyz_to_mat(quat):
  return Rotation.from_quat(wxyz_to_xyzw(quat)).as_matrix()


def mat_to_wxyz(mat):
  return xyzw_to_wxyz(Rotation.from_matrix(mat).as_quat())
