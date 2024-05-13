import os
import time
from typing import Any, Dict, List

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from ..data_utils import (
  binary_mask_to_rle,
  load_assets_cache,
  load_image,
  load_json,
  write_assets_cache,
)
from ..registry import register_dataset_type
from .base_loader import BOPDatasetLoader

M2MM = 1000.0
MM2M = 1 / M2MM
NUM_CLASSES = 2

# obj id -> short name
SHORT_NAMES = None


def try_image_extensions(path, exts=[".jpg", ".png"], force=True):
  if isinstance(exts, str):
    exts = [exts]

  for ext in exts:
    if os.path.exists(path + ext):
      return path + ext
  if force:
    raise ValueError(f"Image not found with these exts: {exts}, path: {path}")
  return None


@register_dataset_type("icbin")
class ICBinDatasetLoader(BOPDatasetLoader):
  def __init__(
    self,
    name: str,
    path: str,
    models_info_path: str,
    detection_path: str = None,
    test_bop_path: str = None,
    include_ids: List[str] = None,
    cache_path: str = None,
    width: int = 640,
    height: int = 480,
    short_names: List[str] = SHORT_NAMES,
    num_scenes: int = None,
    filter_invalid: bool = True,
    **kwargs,
  ):
    super().__init__(
      name=name,
      path=path,
      models_info_path=models_info_path,
      detection_path=detection_path,
      test_bop_path=test_bop_path,
      include_ids=include_ids,
      cache_path=cache_path,
      width=width,
      height=height,
      short_names=short_names,
      num_scenes=num_scenes,
      filter_invalid=filter_invalid,
      **kwargs,
    )

    self._load_models_info()
    # load instances from annotation files
    self._load_dataset_dicts()
    # load instances from detection results from first-stage (yolox)
    self._load_detections()
    # remove unused objects
    self._filter_objects()

  def _load_models_info(self):
    models_info_path = self.models_info_path

    j = load_json(models_info_path)

    short_names = {}
    for key in j.keys():
      obj_id = int(key)
      name = key if self.short_names is None else self.short_names[obj_id]
      short_names[obj_id] = name
    self.short_names = short_names
    self.num_classes = len(self.short_names)

    models_info = {}
    for key, value in j.items():
      obj_id = int(key)
      value["object_id"] = obj_id
      value["short_name"] = self.short_names[obj_id]
      value["diameter"] = value["diameter"] * MM2M
      value["extent"] = [
        value["size_x"] * MM2M,
        value["size_y"] * MM2M,
        value["size_z"] * MM2M,
      ]
      models_info[obj_id] = value

    self.models_info: Dict[int, Any] = models_info

    # temporary category ids
    objid2catid: Dict[int, int] = {}  # to 0-base
    catid2objid: Dict[int, int] = {}

    for cat_id, model_info in enumerate(models_info.values()):
      obj_id = model_info["object_id"]
      objid2catid[obj_id] = cat_id
      catid2objid[cat_id] = obj_id

    self.objid2catid = objid2catid
    self.catid2objid = catid2objid
    if self.include_ids is None:
      self.include_ids = list(self.objid2catid.keys())

  def _load_dataset_dicts(self):
    dataset_dicts = load_assets_cache(self.cache_path)
    if dataset_dicts is not None:
      self.dataset_dicts = dataset_dicts
      return

    start_time = time.time()

    scenes = sorted(os.listdir(self.path))
    if self.num_scenes is None:
      self.num_scenes = len(scenes)
    scenes = scenes[: self.num_scenes]
    dataset_dicts = []

    scene_image_ids = {}
    if self.test_bop_path is not None:
      # load test split info
      targets = load_json(self.test_bop_path)
      for item in targets:
        scene_id = item["scene_id"]
        image_id = item["im_id"]
        if scene_id not in scene_image_ids:
          scene_image_ids[scene_id] = []
        scene_image_ids[scene_id].append(image_id)
      for scene_id, image_ids in scene_image_ids.items():
        scene_image_ids[scene_id] = sorted(image_ids)

    # TODO: add option to disable filtering
    num_invalid_bbox = 0
    num_invalid_mask = 0

    for scene_str in tqdm(scenes, desc="Loading scenes"):
      scene_id = int(scene_str)
      scene_root = os.path.join(self.path, scene_str)

      scene_gt = load_json(os.path.join(scene_root, "scene_gt.json"))
      scene_gt_info = load_json(os.path.join(scene_root, "scene_gt_info.json"))
      scene_camera = load_json(os.path.join(scene_root, "scene_camera.json"))

      image_list = list(scene_gt.keys())

      for image_str in tqdm(image_list, leave=False, desc="Loading annotations"):
        image_id = int(image_str)

        if self.test_bop_path is not None:
          # check if image_id is in the scene_image_ids
          if image_id not in scene_image_ids[scene_id]:
            continue

        # NOTE: train_pbr (.jpg) and train_primesense (.png) use different file extension
        # so we have to test each extension to find the image path.
        rgb_path = try_image_extensions(
          os.path.join(scene_root, f"rgb/{image_id:06d}"), [".jpg", ".png"], force=True
        )
        assert os.path.exists(rgb_path), rgb_path

        depth_path = try_image_extensions(
          os.path.join(scene_root, f"depth/{image_id:06d}"), [".png"], force=False
        )

        scene_image_id = f"{scene_id}/{image_id}"

        K = np.array(scene_camera[image_str]["cam_K"], dtype=np.float32).reshape((3, 3))
        depth_factor = M2MM / scene_camera[image_str]["depth_scale"]

        record = {
          "dataset_name": self.name,
          "file_name": rgb_path,
          "depth_path": depth_path,
          "height": self.height,
          "width": self.width,
          "scene_id": scene_id,
          "image_id": image_id,
          "scene_image_id": scene_image_id,
          "cam": K,
          "depth_factor": depth_factor,
          "scale_factor": M2MM,
        }

        insts = []
        for anno_idx, anno in enumerate(scene_gt[image_str]):
          obj_id = int(anno["obj_id"])  # 1-base
          cat_id = self.objid2catid[obj_id]  # 0-base
          R = np.array(anno["cam_R_m2c"], dtype=np.float32).reshape((3, 3))
          t = np.array(anno["cam_t_m2c"], dtype=np.float32) / M2MM
          quat = Rotation.from_matrix(R).as_quat()  # xyzw
          quat = np.roll(quat, 1, axis=-1)  # xyzw -> wxyz
          tran = t.reshape((3,))

          proj = (record["cam"] @ t.T).T
          proj = proj[:2] / proj[2]

          bbox_visib = scene_gt_info[image_str][anno_idx]["bbox_visib"]
          bbox_obj = scene_gt_info[image_str][anno_idx]["bbox_obj"]
          x1, y1, w, h = bbox_visib
          if self.filter_invalid and (h <= 1 or w <= 1):
            num_invalid_bbox += 1
            continue

          mask_file = os.path.join(
            scene_root, f"mask/{image_id:06d}_{anno_idx:06d}.png"
          )
          mask_visib_file = os.path.join(
            scene_root, f"mask_visib/{image_id:06d}_{anno_idx:06d}.png"
          )
          assert os.path.exists(mask_file), mask_file
          assert os.path.exists(mask_visib_file), mask_visib_file

          mask_single = load_image(mask_visib_file).astype(bool)
          area = mask_single.sum()
          if self.filter_invalid and area <= 64:
            num_invalid_mask += 1
            continue
          mask_rle = binary_mask_to_rle(mask_single, compressed=True)

          mask_full = load_image(mask_file).astype(bool)
          mask_full_rle = binary_mask_to_rle(mask_full, compressed=True)

          visib_fract = scene_gt_info[image_str][anno_idx].get("visib_fract", 1.0)

          inst = {
            "category_id": cat_id,  # 0-base
            "object_id": obj_id,
            "bbox": bbox_visib,
            "bbox_obj": bbox_obj,
            "bbox_mode": "xywh",
            "rotation": quat,  # wxyz
            "translation": tran,  # xyz,
            "centroid_2d": proj,
            "visib_fract": visib_fract,
            "segmentation": mask_rle,
            "mask_full": mask_full_rle,
          }

          model_info = self.models_info[obj_id]
          inst["model_info"] = model_info

          insts.append(inst)
        if len(insts) == 0:
          continue
        record["annotations"] = insts
        dataset_dicts.append(record)

    if num_invalid_mask > 0:
      print(f"Filtered out {num_invalid_mask} instances without valid segmentation.")

    if num_invalid_bbox > 0:
      print(f"Filtered out {num_invalid_bbox} instances without valid box.")

    loaded_time = time.time() - start_time
    print(f"Loaded {len(dataset_dicts)} dataset dicts, using {loaded_time}s")

    write_assets_cache(
      self.cache_path,
      dataset_dicts,
    )
    self.dataset_dicts = dataset_dicts

  def _filter_objects(self):
    # filter out unused annotations
    num_invalid_anno = 0

    new_dataset_dicts = []
    for dataset_dict in self.dataset_dicts:
      annotations = []
      for anno in dataset_dict["annotations"]:
        if anno["object_id"] not in self.include_ids:
          continue
        annotations.append(anno)
      if len(annotations) == 0:
        num_invalid_anno += 1
        continue
      dataset_dict["annotations"] = annotations
      new_dataset_dicts.append(dataset_dict)

    if num_invalid_anno > 0:
      print(f"Filtered out {num_invalid_anno} dataset dicts due to no instances")

    print(
      f"Loaded {len(new_dataset_dicts)} dataset dicts, after filtering out unused objects"
    )

    self.dataset_dicts = new_dataset_dicts
