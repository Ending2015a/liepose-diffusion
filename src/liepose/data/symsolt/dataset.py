import copy
import os
from typing import List

import imageio
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader, Dataset

from liepose.utils import nest

from ..utils import load_file, repeat_dataloader, seed_worker

SHAPE_NAMES = ["tet", "cube", "icosa", "cone", "cyl"]

DATASET_LIST = {}


def register_dataset(dataset_name: str, dataset_path: str):
  """Register dataset
  e.g.
  ```
  register_dataset('train-10k', '/path/to/train-10k/')
  # equivelant
  SymsoltDataset.register_dataset('train-20k', '/path/to/train-20k/')
  dataset = SymsoltDataset('train-10k')
  ```

  Args:
    dataset_name (str):
    dataset_path (str):
  """
  global DATASET_LIST
  assert (
    dataset_name not in DATASET_LIST.keys()
  ), f"The dataset '{dataset_name}' is registered."
  DATASET_LIST[dataset_name] = dataset_path


def create_dataset(
  dataset_name: str,
  shape_names: List[str],
  color_mode: str = "none",
  num_samples: int = None,
  include_equivalents: bool = False,
  pad_equivalents: bool = False,
  **kwargs,
):
  labels = [SHAPE_NAMES.index(shape_name) for shape_name in shape_names]
  return SymsoltDataset(
    dataset_name,
    include_labels=labels,
    color_mode=color_mode,
    num_samples=num_samples,
    include_equivalents=include_equivalents,
    pad_equivalents=pad_equivalents,
    **kwargs,
  )


def collate_fn(batch):
  def stack(xs):
    return np.stack(xs, axis=0)

  return nest.map_nested_tuple(tuple(batch), op=stack)


class SymsoltDataset(Dataset):
  def __init__(
    self,
    dataset_name: str,
    include_labels: List[int] = None,  # shape labels [0~4]
    color_mode: str = "none",  # color, none
    num_samples: int = None,
    num_points: int = None,
    include_equivalents: bool = False,
    pad_equivalents: bool = False,
  ):
    assert (
      dataset_name in DATASET_LIST.keys()
    ), f"Dataset '{dataset_name}' not registered."
    self.path = path = DATASET_LIST[dataset_name]
    self.color_mode = color_mode.lower()
    assert self.color_mode in ["color", "none"]

    anno_path = os.path.join(path, "annotations.gz")
    assert os.path.isfile(anno_path), anno_path

    self.anno_path = anno_path

    # load annotations
    annos = load_file(self.anno_path)

    if include_labels is not None:
      annos = [anno for anno in annos if anno["label_shape"] in include_labels]

    if num_samples is not None:
      num_samples = min(num_samples, len(annos))

    annos = annos[:num_samples]
    self.annos = annos

    # load point clouds
    self.num_points = num_points
    self.points = {}
    max_equivalents = 0
    for anno in annos:
      # cache point clouds
      self.get_point_cloud(anno["points"])
      max_equivalents = max(max_equivalents, anno["num_equivalents"])

    self.include_equivalents = include_equivalents
    self.pad_equivalents = pad_equivalents
    self.max_equivalents = max_equivalents

  def get_point_cloud(self, path):
    if path not in self.points.keys():
      # load from point cloud file
      pcd_path = os.path.join(self.path, path)
      print(f"Loading point cloud from: {pcd_path}")
      pcd = o3d.io.read_point_cloud(pcd_path)
      points = np.array(pcd.points, dtype=np.float32)
      num_points = self.num_points
      if num_points is not None:
        num_points = min(num_points, len(points))
      # cache point cloud
      points = points[:num_points]
      self.points[path] = points

    return self.points[path]

  def __len__(self):
    return len(self.annos)

  def __getitem__(self, index):
    anno = self.annos[index]

    if self.color_mode == "none":
      image_filename = anno["image"]
    else:
      image_filename = anno[f"image_{self.color_mode}"]
    points_filenames = anno["points"]  # {shape}/points.ply
    rotation = anno["rotation"]
    rotations_equivalent = anno["rotations_equivalent"]
    translation = anno["translation"]
    num_equivalents = anno["num_equivalents"]
    label_shape = anno["label_shape"]

    # restore rotation (quat-wxyz) and translation (world-xyz)
    rotation = np.array(rotation, dtype=np.float32).reshape((4,))
    rotations_equivalent = np.array(rotations_equivalent, dtype=np.float32).reshape(
      (num_equivalents, 4)
    )
    translation = np.array(translation, dtype=np.float32).reshape((3,))
    image_path = os.path.join(self.path, image_filename)
    # load image
    image = np.asarray(imageio.imread(image_path)).astype(np.float32)
    # load point clouds
    points = self.get_point_cloud(points_filenames)

    d = {
      "image": image,
      "rotation": rotation,  # wxyz
      "translation": translation,
      "label_shape": label_shape,
      "points": copy.deepcopy(points),
      "index": index,
    }

    if self.include_equivalents:
      # pad equivalents
      # for symmetry loss calculation
      if self.pad_equivalents:
        # quaternion identity (1, 0, 0, 0)
        pad_num = self.max_equivalents - num_equivalents
        if pad_num > 0:
          first = np.array([rotation] * pad_num, dtype=np.float32)
          rotations_equivalent = np.concatenate((rotations_equivalent, first), axis=0)
        d["num_equivalents"] = num_equivalents
      d["rotations_equivalent"] = rotations_equivalent

    return d

  def create_dataloader(self, **kwargs) -> DataLoader:
    if "collate_fn" not in kwargs:
      kwargs["collate_fn"] = collate_fn
    return DataLoader(self, **kwargs)

  def get_batch(self, indices):
    """Manual batching"""
    indices = np.array(indices, dtype=np.int64).flatten()
    batch = []
    for ind in indices:
      batch.append(self[ind])
    return collate_fn(batch)

  @classmethod
  def register_dataset(cls, dataset_name: str, dataset_path: str):
    register_dataset(dataset_name, dataset_path)


class TestDataset:
  def __init__(self, *args, batch_size, **kwargs):
    self.batch_size = batch_size
    self.dataset = create_dataset(*args, **kwargs)

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
  def __init__(self, *args, batch_size, seed, **kwargs):
    self.batch_size = batch_size
    self.seed = seed
    self.dataset = create_dataset(*args, **kwargs)

  def __iter__(self):
    generator = torch.Generator()
    generator.manual_seed(self.seed)

    return repeat_dataloader(
      self.dataset.create_dataloader(
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=generator,
      )
    )

  def __len__(self):
    return None
