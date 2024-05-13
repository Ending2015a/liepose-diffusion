from typing import Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from liepose.utils.npfn import (
  get_allo_rotation,
  get_obj_ray,
  homogeneous,
  wxyz_to_xyzw,
  xyzw_to_wxyz,
)

from ..image import remap


def get_crop_resize_grid(
  center: Tuple[int, int], scale: Tuple[float, float], output_size: Tuple[int, int]
):
  if isinstance(scale, (int, float)):
    scale = (scale, scale)
  if isinstance(output_size, int):
    output_size = (output_size, output_size)
  # convert to [x, y] format, float32
  center = np.array(center, dtype=np.float32)
  scale = np.array(scale, dtype=np.float32)
  output_size = np.array(output_size, dtype=np.float32)
  corner = center - scale / 2
  # create meshgrid
  x = np.arange(output_size[0], dtype=np.float32)
  y = np.arange(output_size[1], dtype=np.float32)
  grid = np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1) / output_size
  grid = grid * scale + corner
  return grid


def crop_resize_intrinsics(
  cam: np.ndarray,
  center: np.ndarray,
  size: Union[float, np.ndarray],
  output_size: np.ndarray,
) -> np.ndarray:
  """Calculate the new camera intrinsics after cropping and resizing the image

  Args:
    cam (np.ndarray): original camera intrinsics (3, 3)
    center (np.ndarray): center position of the crop, (2,), in pixel
    size (np.ndarray): size of the crop before resizing [w, h], (2,), in pixel
    output_size(np.ndarray): size of the crop after resizing [w, h], (2,), in pixel
  """
  crop_xy = center - size / 2
  focal_ratio = output_size / size

  new_cam = cam.copy()
  new_cam[[0, 1], 2] = cam[[0, 1], 2] - crop_xy
  new_cam[[0, 1]] = new_cam[[0, 1]] * focal_ratio[..., np.newaxis]
  return new_cam


class CropResizeRemapper:
  def __init__(
    self,
    cam: np.ndarray,
    center: Tuple[float, float],
    size: Tuple[float, float],
    output_size: Tuple[int, int],
  ):
    """Equivalent to CropResizeAffiner but implemented with remap function
    (slower than warpAffine)
    """
    if isinstance(size, (int, float)):
      size = (size, size)
    if isinstance(output_size, int):
      output_size = (output_size, output_size)

    self.cam = np.asarray(cam, dtype=np.float32)
    self.center = np.asarray(center, dtype=np.float32)
    self.size = np.asarray(size, dtype=np.float32)
    self.output_size = np.asarray(output_size, dtype=np.float32)

    self.focal_ratio = self.output_size / self.size
    self.principle_point = (
      cam[[0, 1], 2] - self.center + self.size / 2
    ) * self.focal_ratio

    self.grid = get_crop_resize_grid(self.center, self.size, self.output_size)
    self.transformed_cam = crop_resize_intrinsics(
      self.cam, self.center, self.size, self.output_size
    )

  def transform_image(self, image: np.ndarray, interp: str = "linear"):
    return remap(image, self.grid, interp)

  def transform_pixels(self, pixels: np.ndarray):
    return pixels

  def transform_pose(self, quat, tran):
    # quat: wxyz
    # tran: xyz
    # convert to allo rotation
    rot = Rotation.from_quat(wxyz_to_xyzw(quat))
    allo_quat = get_allo_rotation(tran)  # wxyz
    allo_rot = Rotation.from_quat(wxyz_to_xyzw(allo_quat)) * rot
    allo_quat = xyzw_to_wxyz(allo_rot.as_quat()).astype(np.float32)
    return allo_quat, tran

  @classmethod
  def inv_transform_pose(cls, quat, tran, transformed_cam=None):
    allo_rot = Rotation.from_quat(wxyz_to_xyzw(quat))
    allo_quat = get_allo_rotation(tran)
    rot = Rotation.from_quat(wxyz_to_xyzw(allo_quat)).inv() * allo_rot
    quat = xyzw_to_wxyz(rot.as_quat())
    return quat, tran


def reproject_points(
  points: np.ndarray, cam: np.ndarray, reproj_cam: np.ndarray, rot: np.ndarray
):
  """Return reprojected points to principle point"""
  assert points.shape[-1] == 2, points.shape
  assert len(points.shape) == 2, points.shape
  points = homogeneous(points)
  points = homogeneous(points @ np.linalg.inv(cam).T)
  points = homogeneous(points @ (reproj_cam @ rot).T)
  return points[..., :2]


def get_bound(center, scale):
  # [left, top, right, down]
  cx, cy = center
  w, h = scale
  bound = np.array(
    [
      [cx - w / 2, cy],  # left
      [cx, cy - h / 2],  # top
      [cx + w / 2, cy],  # right
      [cx, cy + h / 2],  # down
    ],
    dtype=np.float32,
  )
  return bound


def get_principle_rot(cam, point):
  """Return rotation matrix from point in 2D to principle point
  in 3D homogeneous camera space"""
  obj_ray = get_obj_ray(cam, point)
  # rotation from arbitrary point to principle point
  quat = wxyz_to_xyzw(get_allo_rotation(obj_ray))
  rot = Rotation.from_quat(quat).as_matrix()
  return rot


def get_reproject_grid(
  cam: np.ndarray, reproj_cam: np.ndarray, rot: np.ndarray, output_size: np.ndarray
):
  assert len(output_size) == 2, output_size
  x = np.arange(output_size[0], dtype=np.float32)
  y = np.arange(output_size[1], dtype=np.float32)
  grid = np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1)
  grid_shape = grid.shape[:2]
  grid = grid.reshape((-1, 2))
  grid = reproject_points(grid, reproj_cam, cam, rot.T)
  grid = grid.reshape((grid_shape[0], grid_shape[1], 2))
  return grid


class ReprojectionRemapper:
  def __init__(
    self,
    cam: np.ndarray,
    center: Tuple[float, float],
    size: Tuple[float, float],
    output_size: Tuple[int, int],
  ):
    if isinstance(size, (int, float)):
      size = (size, size)
    if isinstance(output_size, int):
      output_size = (output_size, output_size)

    self.cam = np.asarray(cam, dtype=np.float32)
    self.center = np.asarray(center, dtype=np.float32)
    self.size = np.asarray(size, dtype=np.float32)
    self.output_size = np.asarray(output_size, dtype=np.float32)

    # get rotation from bbox center to principle point
    self.ego2allo = get_principle_rot(self.cam, self.center)
    # calculate reprojected bbox size
    reproj_bound = reproject_points(
      get_bound(self.center, self.size), self.cam, self.cam, self.ego2allo
    )
    h_size = np.linalg.norm(reproj_bound[0] - reproj_bound[2], axis=-1)
    v_size = np.linalg.norm(reproj_bound[1] - reproj_bound[3], axis=-1)
    reproj_size = max(h_size, v_size)
    self.reproj_size = np.array((reproj_size, reproj_size), dtype=np.float32)
    self.focal_ratio = self.output_size / self.reproj_size
    self.principle_point = self.output_size / 2

    self.transformed_cam = np.array(
      [
        [self.cam[0, 0] * self.focal_ratio[0], 0, self.principle_point[0]],
        [0, self.cam[1, 1] * self.focal_ratio[1], self.principle_point[1]],
        [0, 0, 1],
      ],
      dtype=np.float32,
    )

    self.grid = get_reproject_grid(
      self.cam, self.transformed_cam, self.ego2allo, self.output_size
    )

  def transform_image(self, image: np.ndarray, interp: str = "linear"):
    return remap(image, self.grid, interp)

  def transform_pixels(self, pixels: np.ndarray):
    pixels = np.asarray(pixels, dtype=np.float32)
    assert pixels.shape[-1] == 2, pixels.shape
    orig_shape = pixels.shape
    pixels = pixels.reshape((-1, 2))
    pixels = reproject_points(pixels, self.cam, self.transformed_cam, self.ego2allo)
    pixels = pixels.reshape(orig_shape)
    return pixels

  def transform_pose(self, quat, tran):
    # quat: wxyz
    # tran: xyz
    rot = Rotation.from_quat(wxyz_to_xyzw(quat)).as_matrix()
    allo_rot = self.ego2allo @ rot
    allo_quat = Rotation.from_matrix(allo_rot).as_quat()
    allo_quat = xyzw_to_wxyz(allo_quat)
    allo_tran = tran @ self.ego2allo.T
    return allo_quat, allo_tran

  def inv_transform_pixels(self, pixels: np.ndarray):
    pixels = np.asarray(pixels, dtype=np.float32)
    assert pixels.shape[-1] == 2, pixels.shape
    orig_shape = pixels.shape
    pixels = pixels.reshape((-1, 2))
    pixels = reproject_points(pixels, self.transformed_cam, self.cam, self.ego2allo.T)
    pixels = pixels.reshape(orig_shape)
    return pixels
