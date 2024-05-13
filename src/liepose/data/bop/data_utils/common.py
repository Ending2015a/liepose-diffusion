import hashlib
import itertools
import json
import logging
import os

import numpy as np
import pycocotools.mask as cocomask

from liepose.utils import workdir

CACHE_ROOT_PATH = workdir(".caches", default="/tmp")


def load_json(path, assert_exist=True):
  if assert_exist:
    assert os.path.isfile(path), path
  else:
    if not os.path.isfile(path):
      return None
  with open(path) as f:
    return json.load(f)


def set_cache_root_path(path):
  global CACHE_ROOT_PATH
  CACHE_ROOT_PATH = path


def get_cache_root_path():
  global CACHE_ROOT_PATH
  return os.environ.get("CACHE_ROOT_PATH", CACHE_ROOT_PATH)


def get_default_cache_path(name, assets_type: str = "assets", hash_info={}):
  hash_name = hashlib.md5(
    (assets_type + "-".join([str(info) for info in hash_info])).encode("utf-8")
  ).hexdigest()
  root_path = get_cache_root_path()
  return os.path.join(root_path, name, f"{assets_type}_{hash_name}.gz")


def load_assets_cache(path):
  from ...utils import load_file

  if not os.path.isfile(path):
    logging.info(f"Cache not found: {path}")
    return None

  try:
    logging.info(f"Loading cache from: {path}")
    assets = load_file(path)
    return assets
  except Exception as e:
    logging.error(f"Failed to load cache: {path}")
    logging.error(f"Some errors occurred: {type(e)}: {str(e)}")
    return None


def write_assets_cache(path, assets):
  from ...utils import write_file

  os.makedirs(os.path.dirname(path), exist_ok=True)
  write_file(path, assets)
  logging.info(f"Writing cache to: {path}")


def binary_mask_to_rle(mask, compressed=True):
  # copied from gdrnpp
  """Encode bianry mask using Run-length encoding"""
  assert mask.ndim == 2, mask.shape
  mask = mask.astype(np.uint8)
  if compressed:
    rle = cocomask.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")
  else:
    rle = {"counts": [], "size": list(mask.shape)}
    counts = rle.get("counts")
    for i, (value, elements) in enumerate(itertools.groupby(mask.ravel(order="F"))):  # noqa: E501
      if i == 0 and value == 1:
        counts.append(0)
      counts.append(len(list(elements)))
  return rle


def rle2mask(rle, height, width):
  # copied from gdrnpp
  if "counts" in rle and isinstance(rle["counts"], list):
    # if compact RLE, ignore this conversion
    # Magic RLE format handling painfully discovered by looking at the
    # COCO API showAnns function.
    rle = cocomask.frPyObjects(rle, height, width)
  mask = cocomask.decode(rle)
  return mask


def bbox_convert_format(bbox, from_mode, to_mode):
  if from_mode == "xywh" and to_mode == "xyxy":
    bbox[..., 2] += bbox[..., 0]
    bbox[..., 3] += bbox[..., 1]
  elif from_mode == "xyxy" and to_mode == "xywh":
    bbox[..., 2] -= bbox[..., 0]
    bbox[..., 3] -= bbox[..., 1]
  else:
    raise NotImplementedError(f"from mode '{from_mode}' to mode '{to_mode}'")
  return bbox


def calc_xyz_from_depth(
  depth: np.ndarray, rot: np.ndarray, tran: np.ndarray, cam: np.ndarray
) -> np.ndarray:
  """This function calculate the pixel-wise xyz value in object's coordinate
  from depth image

  Args:
    depth (np.ndarray): depth image (h, w)
    rot (np.ndarray): rotation matrix of the object (3, 3)
    tran (np.ndarray): translation pf the object (3,)
    cam (np.ndarray): camera intrinsics (3, 3)

  Returns:
    np.ndarray: pixel-wise xyz value (h, w, 3)
  """
  assert len(depth.shape) == 2, depth.shape
  height, width = depth.shape
  y, x = np.meshgrid(
    np.arange(height, dtype=np.float32),
    np.arange(width, dtype=np.float32),
    indexing="ij",
  )
  # pinhole projection
  x = (x - cam[0, 2]) / cam[0, 0] * depth
  y = (y - cam[1, 2]) / cam[1, 1] * depth
  z = depth
  xyz = np.stack((x, y, z), axis=-1)
  # transform to object's coordinate
  # NOTE:
  #    rot_inv = rot.T
  #    xyz @ rot_inv.T
  # => xyz @ rot
  xyz = (xyz - tran) @ rot
  return xyz
