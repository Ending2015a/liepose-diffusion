import logging
import os
from dataclasses import dataclass

import imgaug.augmenters as iaa
import numpy as np

from liepose.utils import datadir

from .common import get_default_cache_path, load_assets_cache, write_assets_cache
from .image import load_image, resize


@dataclass
class Augment:
  random_color_prob: float = 0.8
  random_bg_prob: float = 0.5
  random_bbox_dzi: bool = True
  dzi_pad_scale: float = 1.5
  dzi_scale_ratio: float = 0.25
  dzi_shift_ratio: float = 0.25
  dzi_type: str = "uniform"


def build_color_augment():
  seq = iaa.Sequential(
    [
      iaa.Sometimes(0.5, iaa.CoarseDropout(p=0.2, size_percent=0.05)),
      iaa.Sometimes(0.4, iaa.GaussianBlur((0.0, 3.0))),
      iaa.Sometimes(0.3, iaa.pillike.EnhanceSharpness(factor=(0.0, 50.0))),
      iaa.Sometimes(0.3, iaa.pillike.EnhanceContrast(factor=(0.2, 50.0))),
      iaa.Sometimes(0.5, iaa.pillike.EnhanceBrightness(factor=(0.1, 6.0))),
      iaa.Sometimes(0.3, iaa.pillike.EnhanceColor(factor=(0.0, 20.0))),
      iaa.Sometimes(0.5, iaa.Add((-25, 25), per_channel=0.3)),
      iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
      iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
      iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
      iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=10, per_channel=True)),
      iaa.Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),
      iaa.Sometimes(0.5, iaa.Grayscale(alpha=(0.0, 1.0))),
    ],
    random_order=True,
  )
  pipe = iaa.WithColorspace(from_colorspace="RGB", to_colorspace="RGB", children=seq)
  return pipe


# ===== Background augmentation =====


def get_default_voc_path():
  return datadir("VOCdevkit/VOC2012/", "./")


def load_voc_table_paths(
  voc_path: str = None,
  num_images: int = 10000,
):
  if voc_path is None:
    voc_path = get_default_voc_path()
  cache_path = get_default_cache_path(
    "voc2012_table", "bg_img_paths", dict(voc_path=voc_path, num_images=num_images)
  )
  img_paths = load_assets_cache(cache_path)
  if img_paths is not None:
    return img_paths

  # load VOC images
  voc_set_dir = os.path.join(voc_path, "ImageSets/Main")
  voc_bg_list_path = os.path.join(voc_set_dir, "diningtable_trainval.txt")
  with open(voc_bg_list_path) as f:
    voc_bg_list = [
      line.strip("\r\n").split()[0]
      for line in f.readlines()
      if line.strip("\r\n").split()[1] == "1"
    ]
  img_paths = [
    os.path.join(voc_path, f"JPEGImages/{bg_idx}.jpg") for bg_idx in voc_bg_list
  ]
  assert len(img_paths) > 0, "No background images were found"

  num_images = min(len(img_paths), num_images)
  indices = list(range(len(img_paths)))
  indices = np.random.choice(indices, num_images)
  img_paths = [img_paths[idx] for idx in indices]

  write_assets_cache(cache_path, img_paths)
  logging.info(f"Number of background images: {len(img_paths)}")
  return img_paths


def load_background_image(filename: str, height: int, width: int):
  bg_image = load_image(filename)
  bg_height, bg_width = bg_image.shape[:2]

  # resize and keep aspect ratio
  scale = min(height / bg_height, width / bg_width)
  tar_height = np.ceil(bg_height * scale).astype(np.int32)
  tar_width = np.ceil(bg_width * scale).astype(np.int32)
  bg_image_pil = resize(bg_image, (tar_width, tar_height), pil=True)

  # crop center
  left = max((tar_width - width) // 2, 0)
  top = max((tar_height - height) // 2, 0)
  bg_image_pil = bg_image_pil.crop((left, top, left + width, top + height))

  bg_image = np.asarray(bg_image_pil).astype(np.uint8)
  res_height, res_width = bg_image.shape[:2]
  assert res_height == height and res_width == width
  return bg_image


def build_background_augment(
  voc_path: str = None,
  num_images: int = 10000,
):
  enabled = True
  img_paths = []

  def background_augment(image, mask):
    if (not enabled) or len(img_paths) == 0:
      return image
    height, width = image.shape[:2]
    # select random background
    ind = np.random.randint(0, len(img_paths))
    bg_path = img_paths[ind]
    bg_image = load_background_image(bg_path, height, width)
    # replace background
    image = image.copy().astype(np.uint8)
    mask = mask.astype(np.bool_)
    mask_bg = ~mask
    image[mask_bg] = bg_image[mask_bg]
    return image

  try:
    img_paths = load_voc_table_paths(voc_path, num_images)
  except Exception as e:
    logging.warning(f"Background augment disabled: {type(e).__name__}: {str(e)}")
    enabled = False
  return background_augment


def build_bbox_dzi_augment(
  dzi_pad_scale: float = 1.5,
  dzi_scale_ratio: float = 0.25,
  dzi_shift_ratio: float = 0.25,
  dzi_type: str = "uniform",
):
  def bbox_dzi_augment(bbox, height, width):
    # copied from gdrnpp
    x1, y1, x2, y2 = bbox.copy()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1
    if dzi_type == "uniform":
      scale_ratio = 1 + dzi_scale_ratio * (
        2 * np.random.random_sample() - 1
      )  # scale x0.75 ~ 1.25
      shift_ratio = dzi_shift_ratio * (
        2 * np.random.random_sample(2) - 1
      )  # shift -0.25~0.25
      bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])
      scale = max(bh, bw) * scale_ratio * dzi_pad_scale  # scale x1.125 ~ 1.875
    elif dzi_type == "roi10d":
      _a = -0.15
      _b = 0.15
      x1 += bw * (np.random.rand() * (_b - _a) + _a)
      x2 += bw * (np.random.rand() * (_b - _a) + _a)
      y1 += bh * (np.random.rand() * (_b - _a) + _a)
      y2 += bh * (np.random.rand() * (_b - _a) + _a)
      x1 = min(max(x1, 0), width)
      x2 = min(max(x1, 0), width)
      y1 = min(max(y1, 0), height)
      y2 = min(max(y2, 0), height)
      bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
      scale = max(y2 - y1, x2 - x1) * dzi_pad_scale
    else:
      bbox_center = np.array([cx, cy])
      scale = max(y2 - y1, x2 - x1)
    bbox_center = bbox_center.astype(np.float32)
    scale = min(scale, max(height, width)) * 1.0
    return bbox_center, scale

  return bbox_dzi_augment
