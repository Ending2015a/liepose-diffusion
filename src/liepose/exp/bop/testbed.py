import os
from typing import Callable

import jax
import jax.numpy as jnp
from jaxlie import SO3

from liepose.data.bop.dataset import TestDataset, TrainDataset
from liepose.data.bop.evaluator import Evaluator
from liepose.utils import ops
from liepose.utils.exploration import Options

from ..model import create_model_fn
from ..testbed import Testbed as BaseTestbed


class Testbed(BaseTestbed):
  def create_model(self):
    lie_metrics = self.lie_metrics
    size = self.a.image_res
    ch = 3 + int(self.a.use_coord) * 2  # rgb + xy-coord
    repr_size = lie_metrics.get_repr_size(self.a.repr_type)
    model_fn = create_model_fn(
      out_dim=repr_size,
      in_dim=repr_size,
      image_shape=[1, size, size, ch],
      resnet_depth=self.a.resnet_depth,
      mlp_layers=self.a.mlp_layers,
      fourier_block=self.a.fourier_block,
      activ_fn=self.a.activ_fn,
    )
    model, params = model_fn(jax.random.PRNGKey(self.a.seed))
    return model, params

  # --- test ---

  def create_test_dataset(self):
    return TestDataset(
      self.a.test_datasets,
      batch_size=self.opt.batch_size,
      include_ids=self.a.include_ids,
      input_res=self.a.image_res,
      is_train=False,
      tran_scale=self.a.tran_scale,
      tran_offset=self.a.tran_offset,
      use_reproj=self.a.use_reproj,
      with_coord=self.a.use_coord,
    )

  def create_test_evaluator(self, dataset):
    return Evaluator(
      a=self.a,
      path=self.opt_path,
      exp=self,
      dataset=dataset.dataset,
      full_eval=self.a.full_eval,
    )

  def create_sample_test_fn(self) -> Callable:
    Lie = self.Lie
    lie_metrics = self.lie_metrics

    def sample_test_fn(seed):
      rt = Lie.sample_uniform(seed)
      rt = lie_metrics.as_repr(rt, self.a.repr_type)
      return {"rt": rt}

    return sample_test_fn

  def create_slices_test_fn(self) -> Callable:
    _sample_test_fn = self.create_sample_test_fn()

    def slices_test_fn(seeds, img):
      b = jax.vmap(lambda s: _sample_test_fn(s))(seeds)
      # pad slice dim
      b["img"] = img[jnp.newaxis] / 127.5 - 1.0
      return b

    return slices_test_fn

  def create_get_batch_test_fn(self) -> Callable:
    _slices_test_fn = self.create_slices_test_fn()
    _jit_slices_test_fn = jax.jit(jax.vmap(_slices_test_fn))

    def get_batch_test(seed, batch, n_slices=256):
      img = jnp.array(batch["image"], dtype=jnp.float32)
      if self.a.use_coord:
        coord = jnp.array(batch["coord"], dtype=jnp.float32)
        img = jnp.concatenate((img, coord), axis=-1)
      if self.a.use_mask:
        mask = jnp.array(batch["mask"], dtype=jnp.float32)
        img = img * mask

      batch_size = img.shape[0]
      seeds = jax.random.split(seed, batch_size * n_slices)
      seeds = seeds.reshape((batch_size, n_slices, *seeds.shape[1:]))
      b = _jit_slices_test_fn(seeds, img)
      return b

    return get_batch_test

  # --- eval ---

  def run_eval(self, opt: Options, opt_path: str):
    noise_schedule = self.noise_schedule

    timesteps = noise_schedule.timesteps
    sample_steps = opt.steps

    if sample_steps > timesteps:
      # skip
      return

    os.makedirs(opt_path, exist_ok=True)

    self.opt = opt
    self.opt_path = opt_path

    # create dataset & evaluator
    test_dataset = self.create_test_dataset()
    test_evaluator = self.create_test_evaluator(test_dataset)

    # the evaluation results are saved to opt_path
    test_evaluator.evaluate()

  # --- train ---

  def create_train_dataset(self):
    return TrainDataset(
      self.a.train_datasets,
      batch_size=self.a.batch_size,
      seed=self.a.seed,
      num_workers=self.a.num_workers,
      include_ids=self.a.include_ids,
      input_res=self.a.image_res,
      is_train=True,
      tran_scale=self.a.tran_scale,
      tran_offset=self.a.tran_offset,
      use_reproj=self.a.use_reproj,
      with_coord=self.a.use_coord,
      augment=dict(
        random_color_prob=0.8,
        random_bg_prob=0.5,
        random_bbox_dzi=True,
        dzi_pad_scale=1.5,
        dzi_scale_ratio=0.25,
        dzi_shift_ratio=0.25,
        dzi_type="uniform",
      ),
      equivalent_rots=False,
    )

  def create_sample_train_fn(self):
    Lie = self.Lie
    LieDist = self.LieDist
    lie_metrics = self.lie_metrics
    noise_schedule = self.noise_schedule

    def sample_train_fn(seed, rot, tran):
      # rot: wxyz
      # tran: xyz
      key1, key2 = jax.random.split(seed, 2)
      t = jax.random.randint(
        key2, shape=(), dtype=jnp.int32, minval=0.0, maxval=noise_schedule.timesteps
      )
      r0 = Lie.from_rotation_and_translation(SO3(rot), tran)
      zt = LieDist._sample_unit(key1, n=())
      rt = ops.add(r0, Lie.exp(noise_schedule.sqrt_alphas[t] * zt))
      rt = lie_metrics.as_repr(rt, self.a.repr_type)
      zt = lie_metrics.as_repr(zt, self.a.repr_type)
      r0 = lie_metrics.as_repr(r0, self.a.repr_type)
      return {"rt": rt, "t": t, "zt": zt, "r0": r0}

    return sample_train_fn

  def create_slices_train_fn(self):
    sample_train_fn = self.create_sample_train_fn()

    def slices_train_fn(seeds, img, rot, tran):
      # sample slices
      b = jax.vmap(lambda s: sample_train_fn(s, rot, tran))(seeds)
      # pad slice dim
      b["img"] = img[jnp.newaxis] / 127.5 - 1.0
      return b

    return slices_train_fn

  def create_get_batch_train_fn(self):
    _slices_train_fn = self.create_slices_train_fn()
    jit_slices_train_fn = jax.jit(jax.vmap(_slices_train_fn))

    def get_batch_train(seed, batch, n_slices=256):
      img = jnp.array(batch["image"], dtype=jnp.float32)
      if self.a.use_coord:
        coord = jnp.array(batch["coord"], dtype=jnp.float32)
        img = jnp.concatenate((img, coord), axis=-1)
      if self.a.use_mask:
        mask = jnp.array(batch["mask"], dtype=jnp.float32)
        img = img * mask

      if self.a.use_allo_rot:
        rot = jnp.array(batch["rotation_allo"], dtype=jnp.float32)
      else:
        rot = jnp.array(batch["rotation"], dtype=jnp.float32)

      if self.a.use_scaled_tran:
        tran = jnp.array(batch["translation_scaled"], dtype=jnp.float32)
      elif self.a.use_allo_tran:
        tran = jnp.array(batch["translation_allo"], dtype=jnp.float32)
      else:
        tran = jnp.array(batch["translation"], dtype=jnp.float32)

      batch_size = img.shape[0]
      seeds = jax.random.split(seed, batch_size * n_slices)
      seeds = seeds.reshape((batch_size, n_slices, *seeds.shape[1:]))
      b = jit_slices_train_fn(seeds, img, rot, tran)
      return b

    return get_batch_train
