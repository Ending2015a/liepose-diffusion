import abc

import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], "GPU")

SHAPE_NAMES = ["tet", "cube", "icosa", "cone", "cyl"]


def create_dataset(shapes, mode="train"):
  shapes = [SHAPE_NAMES.index(shape) for shape in shapes]
  dataset = tfds.load("symmetric_solids", split=mode)
  dataset = dataset.filter(lambda x: tf.reduce_any(tf.equal(x["label_shape"], shapes)))
  anno_key = "rotation" if mode == "train" else "rotations_equivalent"
  dataset = dataset.map(
    lambda example: {
      "image": example["image"],
      "rot_gt": example[anno_key],
      "label": example["label_shape"],
    },
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  return dataset


class BaseDataset(abc.ABC):
  def __init__(self, shape_names, batch_size, seed=None):
    self.shape_names = shape_names
    self.batch_size = batch_size
    self.seed = seed

    self.dataset = self.create_dataset()

  @abc.abstractmethod
  def create_dataset(self):
    raise NotImplementedError

  def __iter__(self):
    return iter(self.dataset)

  def __len__(self):
    return None


class TrainDataset(BaseDataset):
  def create_dataset(self):
    dataset = create_dataset(self.shape_names, mode="train")
    dataset = tfds.as_numpy(
      dataset.repeat()
      .shuffle(self.batch_size * 128, seed=self.seed)
      .batch(self.batch_size)
      .prefetch(32)
    )
    return dataset


class TestDataset(BaseDataset):
  def create_dataset(self):
    datasets = [
      create_dataset([shape], mode="test").batch(self.batch_size)
      for shape in self.shape_names
    ]
    assert len(datasets) > 0
    concat_dataset = datasets[0]
    for dataset in datasets[1:]:
      concat_dataset = concat_dataset.concatenate(dataset)

    dataset = tfds.as_numpy(concat_dataset)
    return dataset
