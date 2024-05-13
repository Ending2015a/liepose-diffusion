from typing import Callable

import jax
import jax.numpy as jnp

from liepose.data.symsol.dataset import TestDataset, TrainDataset
from liepose.data.symsol.evaluator import Evaluator
from liepose.utils import ops

from ..testbed import Testbed as BaseTestbed


class Testbed(BaseTestbed):
  # --- test ---

  def create_test_dataset(self):
    return TestDataset(shape_names=self.a.shape_names, batch_size=self.opt.batch_size)

  def create_test_evaluator(self, dataset):
    return Evaluator(self.a, self.opt_path, self)

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

      batch_size = img.shape[0]
      seeds = jax.random.split(seed, batch_size * n_slices)
      seeds = seeds.reshape((batch_size, n_slices, *seeds.shape[1:]))
      b = _jit_slices_test_fn(seeds, img)
      return b

    return get_batch_test

  # --- train ---

  def create_train_dataset(self):
    return TrainDataset(
      shape_names=self.a.shape_names, batch_size=self.a.batch_size, seed=self.a.seed
    )

  def create_sample_train_fn(self):
    Lie = self.Lie
    LieDist = self.LieDist
    lie_metrics = self.lie_metrics
    noise_schedule = self.noise_schedule

    def sample_train_fn(seed, rot):
      # rot: wxyz
      key1, key2 = jax.random.split(seed, 2)
      t = jax.random.randint(
        key2, shape=(), dtype=jnp.int32, minval=0.0, maxval=noise_schedule.timesteps
      )
      r0 = lie_metrics.as_lie(rot)
      zt = LieDist._sample_unit(key1, n=())
      rt = ops.add(r0, Lie.exp(noise_schedule.sqrt_alphas[t] * zt))
      rt = lie_metrics.as_repr(rt, self.a.repr_type)
      zt = lie_metrics.as_repr(zt, self.a.repr_type)
      r0 = lie_metrics.as_repr(r0, self.a.repr_type)
      return {"rt": rt, "t": t, "zt": zt, "r0": r0}

    return sample_train_fn

  def create_slices_train_fn(self):
    sample_train_fn = self.create_sample_train_fn()

    def slices_train_fn(seeds, img, rot):
      # sample slices
      b = jax.vmap(lambda s: sample_train_fn(s, rot))(seeds)
      # pad slice dim
      b["img"] = img[jnp.newaxis] / 127.5 - 1.0
      return b

    return slices_train_fn

  def create_get_batch_train_fn(self):
    _slices_train_fn = self.create_slices_train_fn()
    jit_slices_train_fn = jax.jit(jax.vmap(_slices_train_fn))

    def get_batch_train(seed, batch, n_slices=256):
      img = jnp.array(batch["image"], dtype=jnp.float32)
      rot = jnp.array(batch["rot_gt"], dtype=jnp.float32)

      batch_size = img.shape[0]
      seeds = jax.random.split(seed, batch_size * n_slices)
      seeds = seeds.reshape((batch_size, n_slices, *seeds.shape[1:]))
      b = jit_slices_train_fn(seeds, img, rot)
      return b

    return get_batch_train
