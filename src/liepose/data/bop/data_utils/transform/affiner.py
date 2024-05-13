from typing import Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from liepose.utils.npfn import get_allo_rotation, wxyz_to_xyzw, xyzw_to_wxyz


def get_dir(src_point, rot_rad):
  # copied from gdrnpp
  sn, cs = np.sin(rot_rad), np.cos(rot_rad)

  src_result = [0, 0]
  src_result[0] = src_point[0] * cs - src_point[1] * sn
  src_result[1] = src_point[0] * sn + src_point[1] * cs

  return src_result


def get_3rd_point(a, b):
  # copied from gdrnpp
  direct = a - b
  return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_crop_resize_affine(
  center: Tuple[int, int], scale: Tuple[float, float], output_size: Tuple[int, int]
):
  # copied from gdrnpp
  if isinstance(scale, (int, float)):
    scale = (scale, scale)
  if isinstance(output_size, int):
    output_size = (output_size, output_size)
  rot = 0
  shift = np.array([0, 0], dtype=np.float32)
  center = np.array(center, dtype=np.float32)
  scale = np.array(scale, dtype=np.float32)

  scale_tmp = scale
  src_w = scale_tmp[0]
  dst_w = output_size[0]
  dst_h = output_size[1]

  rot_rad = np.pi * rot / 180
  src_dir = get_dir([0, src_w * -0.5], rot_rad)
  dst_dir = np.array([0, dst_w * -0.5], np.float32)

  src = np.zeros((3, 2), dtype=np.float32)
  dst = np.zeros((3, 2), dtype=np.float32)
  src[0, :] = center + scale_tmp * shift
  src[1, :] = center + src_dir + scale_tmp * shift
  dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
  dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

  src[2:, :] = get_3rd_point(src[0, :], src[1, :])
  dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

  trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

  return trans


def crop_resize_by_warp_affine(
  image: np.ndarray, affine: np.ndarray, output_size: Tuple[int, int], interp="linear"
):
  if isinstance(output_size, int):
    output_size = (output_size, output_size)

  if interp == "linear":
    interp = cv2.INTER_LINEAR
  elif interp == "nearest":
    interp = cv2.INTER_NEAREST
  else:
    raise ValueError(f"Unknown interp method: {interp}")

  image = cv2.warpAffine(
    image, affine, (int(output_size[0]), int(output_size[1])), flags=interp
  )
  return image


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


class CropResizeAffiner:
  def __init__(
    self,
    cam: np.ndarray,
    bbox_center: Tuple[float, float],
    bbox_size: Tuple[float, float],
    output_size: Tuple[int, int],
  ):
    """CropResizeAffiner

    Args:
      cam (np.ndarray): camera intrinsic matrix (3, 3)
      center (Tuple[float, float]): bbox center
      size (Tuple[float, float]): bbox crop size
      output_size (Tuple[float, float]): resized size
    """
    if isinstance(bbox_size, (int, float)):
      bbox_size = (bbox_size, bbox_size)
    if isinstance(output_size, int):
      output_size = (output_size, output_size)

    self.cam = np.asarray(cam, dtype=np.float32)
    self.bbox_center = np.asarray(bbox_center, dtype=np.float32)
    self.bbox_size = np.asarray(bbox_size, dtype=np.float32)
    self.output_size = np.asarray(output_size, dtype=np.int32)

    self.focal_ratio = self.output_size / self.bbox_size
    self.principle_point = (
      cam[[0, 1], 2] - self.bbox_center + self.bbox_size / 2
    ) * self.focal_ratio

    self.affine = get_crop_resize_affine(
      self.bbox_center, self.bbox_size, self.output_size
    )
    self.transformed_cam = crop_resize_intrinsics(
      self.cam, self.bbox_center, self.bbox_size, self.output_size
    )

  def transform_image(self, image: np.ndarray, interp: str = "linear"):
    return crop_resize_by_warp_affine(image, self.affine, self.output_size, interp)

  def transform_pixels(self, pixels: np.ndarray):
    # TODO:
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
