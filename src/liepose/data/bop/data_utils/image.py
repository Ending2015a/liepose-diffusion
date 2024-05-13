from typing import Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from liepose.utils.npfn import homogeneous


def load_image(path, dtype=np.uint8):
  return np.asarray(imageio.imread(path)).astype(dtype)


def resize(
  image: np.ndarray, size: Tuple[int, int], interp: str = "linear", pil: bool = False
):
  if isinstance(size, (int, float)):
    size = (size, size)
  w = int(size[0])
  h = int(size[1])

  if interp == "linear":
    interp = Image.BILINEAR
  else:
    interp = Image.NEAREST

  image = Image.fromarray(image).resize((w, h), interp)

  if not pil:
    image = np.asarray(image, dtype=np.uint8)
  return image


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


def get_affine_transform(
  center: Tuple[int, int], scale: Tuple[float, float], output_size: Tuple[int, int]
):
  # copied from gdrnpp
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
  image: np.ndarray,
  center: Tuple[int, int],
  scale: Tuple[float, float],
  output_size: Tuple[int, int],
  interp="linear",
):
  # copied from gdrnpp
  if isinstance(scale, (int, float)):
    scale = (scale, scale)
  if isinstance(output_size, int):
    output_size = (output_size, output_size)
  trans = get_affine_transform(center, scale, output_size)

  if interp == "linear":
    interp = cv2.INTER_LINEAR
  elif interp == "nearest":
    interp = cv2.INTER_NEAREST
  else:
    raise ValueError(f"Unknown interp method: {interp}")

  orig_ndims = len(image.shape)

  image = cv2.warpAffine(
    image, trans, (int(output_size[0]), int(output_size[1])), flags=interp
  )
  if len(image.shape) < orig_ndims:
    # opencv will reduced the last dim if it is 1
    image = image[..., np.newaxis]
  return image


def get_2d_coord_np(width, height, low=-1, high=1):
  x = np.linspace(low, high, width, dtype=np.float32, endpoint=False)
  y = np.linspace(low, high, height, dtype=np.float32, endpoint=False)
  xy = np.stack(np.meshgrid(x, y), axis=-1)
  return xy


def get_2d_coord_np_proj(cam, width, height):
  x = np.arange(width, dtype=np.float32)
  y = np.arange(height, dtype=np.float32)
  pixels = np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1)
  # to homogeneous
  pixels = homogeneous(pixels)
  pixels = homogeneous(pixels @ np.linalg.inv(cam).T)
  return pixels[..., :2]


def _remap_cpu(
  image: np.ndarray, grid: np.ndarray, method: str = "linear"
) -> np.ndarray:
  h, w, _ = image.shape

  m0 = np.logical_and(grid[..., 0:1] >= 0, grid[..., 0:1] + 1 <= w - 1)
  m1 = np.logical_and(grid[..., 1:2] >= 0, grid[..., 1:2] + 1 <= h - 1)

  ind = np.floor(grid).astype(np.int64)
  r = grid - ind.astype(np.float32)
  x0 = np.clip(ind[..., 0], 0, w - 1)
  y0 = np.clip(ind[..., 1], 0, h - 1)
  x1 = np.clip(x0 + 1, 0, w - 1)
  y1 = np.clip(y0 + 1, 0, h - 1)

  if method == "nearest":
    rx = (0.5 <= r[:, 0:1]).astype(np.float32)
    ry = (0.5 <= r[:, 1:2]).astype(np.float32)
  else:  # linear
    rx = r[:, 0:1]
    ry = r[:, 1:2]

  crop = (
    image[y0, x0] * (1.0 - rx) * (1.0 - ry)
    + image[y0, x1] * rx * (1.0 - ry)
    + image[y1, x0] * (1.0 - rx) * ry
    + image[y1, x1] * rx * ry
  )

  return crop * (m0 * m1)


def _remap_cv2(image: np.ndarray, grid: np.ndarray, method: str = "linear"):
  if method == "linear":
    method = cv2.INTER_LINEAR
  elif method == "nearest":
    method = cv2.INTER_NEAREST
  else:
    raise NotImplementedError(f"unknown method: {method}")

  crop = cv2.remap(image, grid, None, method)
  return crop


_REMAP_CV2_IMPL = False


def remap(image: np.ndarray, grid: np.ndarray, method: str = "linear") -> np.ndarray:
  # preprocess
  image = image.astype(np.float32)
  grid = grid.astype(np.float32)

  if _REMAP_CV2_IMPL:
    # cv2 impl is faster than numpy
    return _remap_cv2(image, grid, method)

  # preprocess
  orig_ndims = len(image.shape)
  h_grid, w_grid, _ = grid.shape
  grid = grid.reshape((-1, 2))

  if len(image.shape) == 2:
    image = image[..., np.newaxis]

  crop = _remap_cpu(image, grid, method)

  # postprocess
  crop = crop.reshape((h_grid, w_grid, -1))
  if orig_ndims == 2:
    crop = crop[..., 0]
  return crop
