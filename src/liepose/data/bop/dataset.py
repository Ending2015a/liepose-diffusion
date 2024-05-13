import copy
from typing import List

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

from liepose.utils import nest
from liepose.utils.npfn import wxyz_to_xyzw, xyzw_to_wxyz

from ..utils import repeat_dataloader, seed_worker
from .data_utils import (
  bbox_convert_format,
  calc_xyz_from_depth,
  get_2d_coord_np_proj,
  load_image,
  rle2mask,
)
from .data_utils.augment import (
  Augment,
  build_background_augment,
  build_bbox_dzi_augment,
  build_color_augment,
)
from .data_utils.transform import CropResizeAffiner, ReprojectionRemapper
from .dataset_loaders import (
  BOPDatasetLoader,
  concat_datasets,
  load_dataset_by_name,
  load_datasets,
)


def collate_fn(batch):
  def stack(xs):
    # TODO: reliable way to test if the data can be stacked
    # using numpy (numerical)
    if isinstance(xs[0], str):
      return xs
    return np.stack(xs, axis=0)

  return nest.map_nested_tuple(tuple(batch), op=stack)


class BOPDataset(Dataset):
  def __init__(
    self,
    datasets: List[str],
    augment: Augment = {},
    input_res: int = 256,
    output_res: int = None,
    is_train: bool = True,
    **kwargs,
  ):
    self.dataset = concat_datasets(load_datasets(datasets, **kwargs))
    self.dataset.reduce_categories()
    self.width = self.dataset.width
    self.height = self.dataset.height
    self.aspect = self.dataset.aspect

    self.augment = Augment(**augment)
    self.input_res = input_res
    self.output_res = output_res or input_res
    self.is_train = is_train

    if self.is_train:
      self.build_augments()

  def build_augments(self):
    self.color_augment = build_color_augment()
    self.background_augment = build_background_augment()

  @classmethod
  def load(cls, dataset_name, **kwargs) -> BOPDatasetLoader:
    return load_dataset_by_name(dataset_name, **kwargs)

  def __getitem__(self, index):
    raise NotImplementedError

  def create_dataloader(self, **kwargs) -> DataLoader:
    if "collate_fn" not in kwargs:
      kwargs["collate_fn"] = collate_fn
    return DataLoader(self, **kwargs)

  def get_batch(self, indices):
    indices = np.array(indices, dtype=np.int64).flatten()
    batch = []
    for ind in indices:
      batch.append(self[ind])
    return collate_fn(batch)


class BOPPoseDataset(BOPDataset):
  def __init__(
    self,
    datasets: List[str],
    equivalent_rots: bool = False,
    with_coord: bool = False,
    with_mask_full: bool = False,
    with_depth: bool = False,
    with_rendered_depth: bool = False,
    with_xyz: bool = False,
    tran_scale: List[float] = [1.0, 1.0, 1.0],
    tran_offset: List[float] = [0.0, 0.0, 0.0],
    use_reproj: bool = False,
    **kwargs,
  ):
    super().__init__(datasets, **kwargs)
    self.dataset.flatten_annotations()
    self.equivalent_rots = equivalent_rots
    self.with_coord = with_coord
    self.with_mask_full = with_mask_full
    self.with_depth = with_depth
    self.with_rendered_depth = with_rendered_depth
    self.with_xyz = with_xyz
    self.tran_scale = tran_scale
    self.tran_offset = tran_offset
    self.use_reproj = use_reproj

  def build_augments(self):
    self.color_augment = build_color_augment()
    self.background_augment = build_background_augment()
    self.bbox_dzi_augment = build_bbox_dzi_augment(
      self.augment.dzi_pad_scale,
      self.augment.dzi_scale_ratio,
      self.augment.dzi_shift_ratio,
      self.augment.dzi_type,
    )

  def __len__(self):
    return len(self.dataset.dataset_dicts)

  def read_data_train(self, dataset_dict, index):
    assert "file_name" in dataset_dict
    image_path = dataset_dict["file_name"]  # load rgb image
    image = load_image(image_path)
    height, width = image.shape[:2]

    if self.with_depth:
      assert "depth_path" in dataset_dict
      depth_path = dataset_dict["depth_path"]
      depth = load_image(depth_path, dtype=np.uint16)
      depth_factor = dataset_dict["depth_factor"]
      assert height == depth.shape[0], depth.shape[0]
      assert width == depth.shape[1], depth.shape[1]
      depth = depth.reshape((height, width, 1))
      depth = (depth / depth_factor).astype(np.float32)

    cam = np.asarray(dataset_dict["cam"], dtype=np.float32)

    # inst_infos, see BOPDatasetLoader.flatten_annotations()
    assert "inst_infos" in dataset_dict
    inst_infos = dataset_dict["inst_infos"]

    assert "model_info" in inst_infos
    model_info = inst_infos["model_info"]

    if "segmentation" in inst_infos:
      mask = rle2mask(inst_infos["segmentation"], height, width)

    if "mask_full" in inst_infos and self.with_mask_full:
      mask_full = rle2mask(inst_infos["mask_full"], height, width)

    object_id = model_info["object_id"]

    if self.with_coord:
      coord_2d = get_2d_coord_np_proj(cam, width, height)

    # random background augmentation
    if np.random.rand() < self.augment.random_bg_prob:
      image = self.background_augment(image, mask)

    # random rgb augmentation
    if np.random.rand() < self.augment.random_color_prob:
      image = self.color_augment.augment_image(image)

    # calculate bbox range
    bbox_mode = inst_infos["bbox_mode"]

    if "bbox_obj" in inst_infos:
      bbox = np.asarray(inst_infos["bbox_obj"], dtype=np.float32)
      bbox = bbox_convert_format(bbox, bbox_mode, "xyxy")
      x1, y1, x2, y2 = bbox
      bbox_xyxy = np.asarray(
        [max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)], dtype=np.float32
      )
    else:
      bbox = np.asarray(inst_infos["bbox"], dtype=np.float32)
      bbox = bbox_convert_format(bbox, bbox_mode, "xyxy")
      bbox_xyxy = bbox

    x1, y1, x2, y2 = bbox_xyxy
    bbox_w = max(x2 - x1, 1)
    bbox_h = max(y2 - y1, 1)

    # random augment bbox
    if self.augment.random_bbox_dzi:
      bbox_center, dzi_size = self.bbox_dzi_augment(bbox_xyxy, self.height, self.width)
    else:
      bbox_center = 0.5 * np.asarray([x1 + x2, y1 + y2], dtype=np.float32)
      dzi_size = max(bbox_w, bbox_h) * self.augment.dzi_pad_scale
      dzi_size = min(dzi_size, max(height, width)) * 1.0

    roi_bbox_size = np.array([bbox_w, bbox_h], dtype=np.float32)

    if self.use_reproj:
      TransformType = ReprojectionRemapper
    else:
      TransformType = CropResizeAffiner

    Fin = TransformType(cam, bbox_center, dzi_size, self.input_res)
    Fout = Fin
    if self.input_res != self.output_res:
      Fout = TransformType(cam, bbox_center, dzi_size, self.output_res)

    # crop image
    image = image.astype(np.float32)
    roi_image = Fin.transform_image(image)

    # crop image coord
    if self.with_coord:
      coord_2d = coord_2d.astype(np.float32)
      roi_coord_2d = Fin.transform_image(coord_2d)

    # crop mask
    if "segmentation" in inst_infos:
      mask = mask.astype(np.float32)
      roi_mask = Fout.transform_image(mask, "nearest")
      if len(roi_mask.shape) == 2:
        roi_mask = np.expand_dims(roi_mask, axis=-1)

    if "mask_full" in inst_infos and self.with_mask_full:
      mask_full = mask_full.astype(np.float32)
      roi_mask_full = Fout.transform_image(mask_full, "nearest")
      if len(roi_mask_full.shape) == 2:
        roi_mask_full = np.expand_dims(roi_mask_full, axis=-1)

    # crop depth
    if self.with_depth:
      depth = depth.astype(np.float32)
      roi_depth = Fout.transform_image(depth, "nearest")  # (h, w, 1)
      if len(roi_depth.shape) == 2:
        roi_depth = np.expand_dims(roi_depth, axis=-1)

    roi_cam = Fout.transformed_cam

    if self.with_rendered_depth:
      # TODO:
      raise NotImplementedError()

    # wxyz
    quat = np.asarray(inst_infos["rotation"], dtype=np.float32)
    tran = np.asarray(inst_infos["translation"], dtype=np.float32)

    if self.with_xyz:
      assert self.with_depth or self.with_rendered_depth
      # TODO: choose which depth to use. True depth or rendered depth
      roi_xyz = calc_xyz_from_depth(
        depth=roi_depth[..., 0],  # (h, w)
        rot=Rotation.from_quat(wxyz_to_xyzw(quat, -1, axis=-1)).as_matrix(),
        tran=tran,
        cam=roi_cam,
      )
      xyz_mask = roi_mask_full if self.with_rendered_depth else roi_mask
      roi_xyz = roi_xyz * xyz_mask.reshape(xyz_mask.shape[:2] + (1,))  # (h, w, 1)

    # wxyz
    quat = np.asarray(inst_infos["rotation"], dtype=np.float32)
    tran = np.asarray(inst_infos["translation"], dtype=np.float32)

    # allocentric rotation and translation
    quat_allo, tran_allo = Fout.transform_pose(quat, tran)

    # only reprojection remapper use this way to calculate the reprojected obj center
    if self.use_reproj:
      # center of the reprojected bbox
      proj_bbox_center = np.array(roi_cam[[0, 1], 2], dtype=np.float32)
      proj_obj_center = tran_allo @ roi_cam.T
      proj_obj_center = proj_obj_center[:2] / proj_obj_center[2]
    else:
      # no reprojection
      proj_bbox_center = bbox_center
      proj_obj_center = inst_infos["centroid_2d"]

    focal_ratio = Fout.focal_ratio[0]
    delta_c = proj_obj_center - proj_bbox_center

    # Dx = (Ox - Cx) / Sdzi
    # Dy = (Oy - Cy) / Sdzi
    # Dz = z * Sdzi / Sinp
    tran_scale = np.asarray(self.tran_scale, dtype=np.float32)
    tran_offset = np.asarray(self.tran_offset, dtype=np.float32)
    tran_ratio = (
      np.asarray([bbox_w, bbox_h, focal_ratio], dtype=np.float32) * tran_scale
    )

    tran_scaled = (
      np.array([delta_c[0], delta_c[1], tran_allo[2]], dtype=np.float32) / tran_ratio
      - tran_offset
    )

    # get equivalent poses
    quat_eqs = []
    if self.equivalent_rots:
      rot = Rotation.from_quat(wxyz_to_xyzw(quat))
      if "symmetries_continuous" in model_info:
        for sym_cont in model_info["symmetries_continuous"]:
          axis = np.asarray(sym_cont["axis"])
          for angle in range(360):
            sym_rot = Rotation.from_rotvec(axis * np.radians(angle))
            quat_eq = xyzw_to_wxyz((rot * sym_rot).as_quat())
            quat_eqs.append(quat_eq)
      if "symmetries_discrerte" in model_info:
        for sym_disc in model_info["symmetries_discrete"]:
          sym_rot = np.asarray(sym_disc, dtype=np.float64).reshape((4, 4))[:3, :3]
          quat_eq = xyzw_to_wxyz((rot * sym_rot).as_quat())  # wxyz
          quat_eqs.append(quat_eq)
    else:
      quat_eqs.append(quat)

    quat_eqs = np.array(quat_eqs, dtype=np.float32)

    output_dict = {
      "index": index,
      "object_id": object_id,
      "image": roi_image,
      "bbox": bbox,
      "cam": cam,
      "cam_crop": roi_cam,
      "width": width,
      "height": height,
      "bbox_size": roi_bbox_size,
      "bbox_center": bbox_center,
      "proj_bbox_center": proj_bbox_center,
      "dzi_size": dzi_size,
      "focal_ratio": focal_ratio,
      "rotation": quat,  # wxyz
      "rotation_allo": quat_allo,  # wxyz
      "rotation_equivalents": quat_eqs,  # wxyz
      "translation": tran,
      "translation_allo": tran_allo,
      "translation_scaled": tran_scaled,
      "translation_ratio": tran_ratio,
      "translation_offset": tran_offset,
    }

    for key, value in dataset_dict.items():
      if key in ["scene_id", "image_id", "scale_factor"]:
        output_dict[key] = value

    for key, value in inst_infos.items():
      if key in ["category_id", "object_id", "time", "score"]:
        output_dict[key] = value

    if self.with_coord:
      output_dict["coord"] = roi_coord_2d

    if self.with_depth:
      output_dict["depth"] = roi_depth

    if self.with_rendered_depth:
      raise NotImplementedError()

    if self.with_xyz:
      output_dict["xyz"] = roi_xyz

    if "segmentation" in inst_infos:
      output_dict["mask"] = roi_mask

    if "mask_full" in inst_infos and self.with_mask_full:
      output_dict["mask_full"] = roi_mask_full

    return output_dict

  def read_data_test(self, dataset_dict, index):
    assert "file_name" in dataset_dict
    image_path = dataset_dict["file_name"]  # load rgb image
    image = load_image(image_path)
    height, width = image.shape[:2]

    cam = np.asarray(dataset_dict["cam"], dtype=np.float32)

    # inst_infos, see BOPDatasetLoader.flatten_annotations()
    assert "inst_infos" in dataset_dict
    inst_infos = dataset_dict["inst_infos"]

    if "segmentation" in inst_infos:
      mask = rle2mask(inst_infos["segmentation"], height, width)

    if "mask_full" in inst_infos and self.with_mask_full:
      mask_full = rle2mask(inst_infos["mask_full"], height, width)

    if self.with_coord:
      coord_2d = get_2d_coord_np_proj(cam, width, height)

    bbox_mode = inst_infos["bbox_mode"]

    if "bbox_est" in inst_infos:
      bbox = np.asarray(inst_infos["bbox_est"], dtype=np.float32)
    elif "bbox_obj" in inst_infos:
      bbox = np.asarray(inst_infos["bbox_obj"], dtype=np.float32)
    else:
      bbox = np.asarray(inst_infos["bbox"], dtype=np.float32)

    bbox = bbox_convert_format(bbox, bbox_mode, "xyxy")
    bbox_xyxy = bbox

    x1, y1, x2, y2 = bbox_xyxy
    bbox_w = max(x2 - x1, 1)
    bbox_h = max(y2 - y1, 1)

    bbox_center = 0.5 * np.asarray([x1 + x2, y1 + y2], dtype=np.float32)
    dzi_size = max(bbox_w, bbox_h) * self.augment.dzi_pad_scale
    dzi_size = min(dzi_size, max(height, width)) * 1.0

    roi_bbox_size = np.array([bbox_w, bbox_h], dtype=np.float32)

    if self.use_reproj:
      TransformType = ReprojectionRemapper
    else:
      TransformType = CropResizeAffiner

    Fin = TransformType(cam, bbox_center, dzi_size, self.input_res)
    Fout = Fin
    if self.input_res != self.output_res:
      Fout = TransformType(cam, bbox_center, dzi_size, self.output_res)

    # crop image
    image = image.astype(np.float32)
    roi_image = Fin.transform_image(image)

    if self.with_coord:
      coord_2d = coord_2d.astype(np.float32)
      roi_coord_2d = Fin.transform_image(coord_2d)

    # crop mask
    if "segmentation" in inst_infos:
      mask = mask.astype(np.float32)
      roi_mask = Fout.transform_image(mask, "nearest")
      if len(roi_mask.shape) == 2:
        roi_mask = np.expand_dims(roi_mask, axis=-1)

    if "mask_full" in inst_infos and self.with_mask_full:
      mask_full = mask_full.astype(np.float32)
      roi_mask_full = Fout.transform_image(mask_full, "nearest")
      if len(roi_mask_full.shape) == 2:
        roi_mask_full = np.expand_dims(roi_mask_full, axis=-1)

    roi_cam = Fout.transformed_cam
    if self.use_reproj:
      proj_bbox_center = np.array(roi_cam[[0, 1], 2], dtype=np.float32)
    else:
      proj_bbox_center = bbox_center

    focal_ratio = Fout.focal_ratio[0]

    tran_scale = np.asarray(self.tran_scale, dtype=np.float32)
    tran_offset = np.asarray(self.tran_offset, dtype=np.float32)
    tran_ratio = (
      np.asarray([bbox_w, bbox_h, focal_ratio], dtype=np.float32) * tran_scale
    )

    output_dict = {
      "index": index,
      "image": roi_image,
      "bbox": bbox,
      "cam": cam,
      "cam_crop": roi_cam,
      "width": width,
      "height": height,
      "bbox_size": roi_bbox_size,
      "bbox_center": bbox_center,
      "proj_bbox_center": proj_bbox_center,
      "dzi_size": dzi_size,
      "focal_ratio": focal_ratio,
      "translation_ratio": tran_ratio,
      "translation_offset": tran_offset,
    }

    for key, value in dataset_dict.items():
      if key in ["scene_id", "image_id", "scale_factor"]:
        output_dict[key] = value

    for key, value in inst_infos.items():
      if key in ["category_id", "object_id", "time", "score"]:
        output_dict[key] = value
      if key in ["rotation", "translation"]:
        output_dict[key] = np.asarray(value, dtype=np.float32)

    if "translation" in inst_infos:
      quat = np.asarray(inst_infos["rotation"], dtype=np.float32)
      tran = np.asarray(inst_infos["translation"], dtype=np.float32)
      quat_allo, tran_allo = Fout.transform_pose(quat, tran)
      if self.use_reproj:
        proj_obj_center = tran_allo @ roi_cam.T
        proj_obj_center = proj_obj_center[:2] / proj_obj_center[2]
      else:
        proj_obj_center = inst_infos["centroid_2d"]

      delta_c = proj_obj_center - proj_bbox_center

      tran_scaled = (
        np.array([delta_c[0], delta_c[1], tran_allo[2]], dtype=np.float32) / tran_ratio
        - tran_offset
      )
      output_dict["rotation_allo"] = quat_allo
      output_dict["translation_scaled"] = tran_scaled

    if self.with_coord:
      output_dict["coord"] = roi_coord_2d

    if "segmentation" in inst_infos:
      output_dict["mask"] = roi_mask

    if "mask_full" in inst_infos and self.with_mask_full:
      output_dict["mask_full"] = roi_mask_full

    return output_dict

  def read_data(self, dataset_dict, index):
    dataset_dict = copy.deepcopy(dataset_dict)
    if self.is_train:
      return self.read_data_train(dataset_dict, index)
    else:
      return self.read_data_test(dataset_dict, index)

  def __getitem__(self, index):
    dataset_dict = self.dataset.dataset_dicts[index]
    return self.read_data(dataset_dict, index)


class TestDataset:
  def __init__(self, *args, batch_size, **kwargs):
    self.batch_size = batch_size
    self.dataset = BOPPoseDataset(*args, **kwargs)

    num = len(self.dataset)
    batches = np.array_split(np.arange(num), np.arange(num, step=self.batch_size))[1:]
    self.batches = batches

  def __iter__(self):
    self.iter = iter(self.batches)
    return self

  def __next__(self):
    indices = next(self.iter)
    batch_data = self.dataset.get_batch(indices)
    return batch_data

  def __len__(self):
    return len(self.batches)


class TrainDataset:
  def __init__(self, *args, batch_size, seed, num_workers, **kwargs):
    self.batch_size = batch_size
    self.seed = seed
    self.num_workers = num_workers
    self.dataset = BOPPoseDataset(*args, **kwargs)

  def __iter__(self):
    generator = torch.Generator()
    generator.manual_seed(self.seed)

    return repeat_dataloader(
      self.dataset.create_dataloader(
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=self.num_workers,
        worker_init_fn=seed_worker,
        drop_last=True,
        generator=generator,
      )
    )

  def __len__(self):
    return None
