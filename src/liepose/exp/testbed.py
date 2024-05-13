import abc
import os
import time
from typing import Any, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from flax.metrics import tensorboard
from flax.training import checkpoints, train_state
from jaxlie import MatrixLieGroup
from tqdm import tqdm

from ..data.utils import set_seed
from ..dist import LieDist
from ..noise.ve import PowerNoiseSchedule
from ..utils import ops, try_restore_state
from ..utils.exploration import Options
from .model import create_model_fn


class TrainState(train_state.TrainState):
  batch_stats: Any
  params_ema: Any = None


def get_vars_dict(state, key: str = None, ema: bool = False):
  params = state.params_ema if ema else state.params
  batch_stats = state.batch_stats
  if key is not None:
    params = params[key]
    batch_stats = batch_stats[key]
  return {"params": params, "batch_stats": batch_stats}


def get_params_dict(state, key: str = None, ema: bool = False):
  params = state.params_ema if ema else state.params
  if key is not None:
    params = params[key]
  return {"params": params}


# def get_params_dict(state, key: str=None, ema: bool = False):
#   params = state.params_ema if ema else state.params
#   batch_stats = state.batch_stats
#   d = {}
#   if key is not None:
#     if key in params:
#       d['params'] = params[key]
#     if key in batch_stats:
#       d['batch_stats'] = batch_stats[key]
#   return d


def save_checkpoint(ckpt_dir: str, target: TrainState, step: Any):
  if jax.process_index() > 0:
    return
  return checkpoints.save_checkpoint(
    ckpt_dir=ckpt_dir, target=target, step=step, keep=float("inf")
  )


def restore_state(ckpt_dir: str, state: TrainState, step: Any) -> TrainState:
  restored_state, restored_path = try_restore_state(
    ckpt_dir=ckpt_dir, state=state, step=step
  )

  if restored_path is None:
    print(f"No checkpoints were found in {ckpt_dir}")
    return state

  if restored_state is None:
    raise ValueError(
      f"Failed to restore model from: {restored_path}\n"
      "This might be caused because the ckpt file you specified does not exist,"
      " or the ckpt file is broken"
    )

  print(f"Restored checkpoints from: {restored_path}")
  del state
  return restored_state


class Testbed(abc.ABC):
  """Lie group"""

  Lie: MatrixLieGroup
  """Lie group distribution"""
  LieDist: LieDist
  """Lie group metrics"""
  lie_metrics: Any

  def __init__(self, a: Options, exp_path: str, setup: bool = True):
    self.a = a
    self.exp_path = exp_path

    self.opt = None
    self.opt_path = None
    self.noise_schedule = None
    self.model = None
    self.optim = None
    self.state = None
    self.lr_schedule = None
    self.num_devices = None
    self.devices = None

    if setup:
      self.setup()

  def setup(self):
    set_seed(self.a.seed)
    self.noise_schedule = self.create_noise_schedule()
    # create model, optimizer and learning rate schedule
    model, params = self.create_model()
    optim, lr_schedule = self.create_optimizer_and_lr()

    self.model = model
    self.optim = optim
    self.lr_schedule = lr_schedule
    self.num_devices = jax.local_device_count()
    self.devices = jax.local_devices()

    # create and restore training states
    state = self.create_state(params, optim)
    state = self.restore_state(state)
    self.state = state

    return self

  def assert_setup(self):
    assert self.model is not None
    assert self.state is not None

  def create_model(self):
    lie_metrics = self.lie_metrics
    size = self.a.image_res
    repr_size = lie_metrics.get_repr_size(self.a.repr_type)
    model_fn = create_model_fn(
      out_dim=repr_size,
      in_dim=repr_size,
      image_shape=[1, size, size, 3],
      resnet_depth=self.a.resnet_depth,
      mlp_layers=self.a.mlp_layers,
      fourier_block=self.a.fourier_block,
      activ_fn=self.a.activ_fn,
    )
    model, params = model_fn(jax.random.PRNGKey(self.a.seed))
    return model, params

  def create_noise_schedule(self):
    """Create noise schedule"""
    noise_type = self.a.noise_type.lower() or "power"
    if noise_type == "power":
      return PowerNoiseSchedule(
        alpha_start=self.a.noise_start,
        alpha_end=self.a.noise_end,
        timesteps=self.a.timesteps,
        power=self.a.power,
      )
    else:
      raise ValueError(f"Unknown noise type: {noise_type}")

  def create_optimizer_and_lr(self):
    tran_steps = int(self.a.train_steps * self.a.lr_decay_steps)
    tran_begin = int(self.a.train_steps * self.a.lr_decay_start)

    lr_schedule = optax.exponential_decay(
      init_value=self.a.init_lr,
      transition_steps=tran_steps,
      decay_rate=self.a.lr_decay_rate,
      transition_begin=tran_begin,
      end_value=self.a.end_lr,
    )
    optim = optax.chain(optax.adamw(learning_rate=lr_schedule))
    return optim, lr_schedule

  def create_state(self, params, optim) -> TrainState:
    assert self.model is not None
    return TrainState.create(
      apply_fn=self.model.apply,
      params=params["params"],
      tx=optim,
      batch_stats=params["batch_stats"],
      params_ema=params["params"],
    )

  def restore_state(self, state) -> TrainState:
    if self.a.restore_path is not None:
      ckpt_dir = self.a.restore_path
    else:
      ckpt_dir = os.path.join(self.exp_path, "ckpt")

    return restore_state(ckpt_dir=ckpt_dir, state=state, step=self.a.restore_step)

  @property
  def global_step(self):
    return self.state.step

  # --- inference ---

  def create_p_sample_ddpm_fn(self) -> Callable:
    Lie = self.Lie
    LieDist = self.LieDist
    lie_metrics = self.lie_metrics
    learn_noise = self.a.learn_noise
    noise_schedule = self.noise_schedule
    beta_skip = self.opt.beta_skip

    def p_sample_ddpm_fn(key, mu, rt, t, tp):
      sa = noise_schedule.sqrt_alphas[t]
      sap = noise_schedule.sqrt_alphas_prev[tp]
      if beta_skip:
        # theoretically correct
        sb = jnp.sqrt(sa**2 - sap**2)
      else:
        sb = noise_schedule.sqrt_betas[t]

      rt = lie_metrics.as_lie(rt)
      if learn_noise:
        zt = lie_metrics.as_tan(mu)
        r0 = ops.lsub(rt, Lie.exp(sa * zt))
      else:
        r0 = lie_metrics.as_lie(mu)
        zt = ops.rsub(r0, rt).log() / sa

      noise = LieDist._sample_unit(key, n=())
      rp = ops.lsub(rt, Lie.exp(sb / sa * sb * zt))
      rp = ops.add(rp, Lie.exp(sap / sa * sb * noise))

      r0 = lie_metrics.as_repr(r0, self.a.repr_type)
      rp = lie_metrics.as_repr(rp, self.a.repr_type)
      return r0, rp

    return p_sample_ddpm_fn

  def create_p_sample_score_fn(self) -> Callable:
    Lie = self.Lie
    LieDist = self.LieDist
    lie_metrics = self.lie_metrics
    learn_noise = self.a.learn_noise
    noise_schedule = self.noise_schedule
    beta_skip = self.opt.beta_skip
    eta = self.opt.eta

    def p_sample_score_fn(key, mu, rt, t, tp):
      sa = noise_schedule.sqrt_alphas[t]
      sap = noise_schedule.sqrt_alphas_prev[tp]
      if beta_skip:
        # theoretically correct
        sb = jnp.sqrt(sa**2 - sap**2)
      else:
        sb = noise_schedule.sqrt_betas[t]

      # estimate r0 from rt using inverse of Eq.7
      rt = lie_metrics.as_lie(rt)
      if learn_noise:
        zt = lie_metrics.as_tan(mu)
        r0 = ops.lsub(rt, Lie.exp(sa * zt))
      else:
        r0 = lie_metrics.as_lie(mu)
        zt = ops.rsub(r0, rt).log() / sa

      # estimate rp using Eq.7 (eta=noise rate)
      if eta > 0.0:
        noise = LieDist._sample_unit(key, n=())
        sig = eta * (sap / sa * sb)
        c = jnp.sqrt(sap**2 - sig**2)
        rp = ops.add(r0, Lie.exp(c * zt))
        rp = ops.add(rp, Lie.exp(sig * noise))
      else:
        # without noise
        rp = ops.add(r0, Lie.exp(sap * zt))

      r0 = lie_metrics.as_repr(r0, self.a.repr_type)
      rp = lie_metrics.as_repr(rp, self.a.repr_type)
      return r0, rp

    return p_sample_score_fn

  def create_p_sample_fn(self) -> Callable:
    if self.opt.use_ddpm:
      _p_sample_fn = self.create_p_sample_ddpm_fn()
    else:
      _p_sample_fn = self.create_p_sample_score_fn()
    return _p_sample_fn

  def create_p_sample_apply(self) -> Callable:
    # create jitted function
    _p_sample_fn = self.create_p_sample_fn()
    _head_apply = self.create_head_apply()
    _jit_p_sample_fn = jax.jit(jax.vmap(_p_sample_fn))

    def _p_sample_apply(seed, feat, rt, t, tp):
      batch_size = rt.shape[0]
      tt = jnp.ones([batch_size, 1], dtype=jnp.float32) * t
      t = jnp.ones([batch_size], dtype=jnp.int32) * t
      tp = jnp.ones([batch_size], dtype=jnp.int32) * tp
      mu = _head_apply(feat, rt, tt)
      return _jit_p_sample_fn(jax.random.split(seed, batch_size), mu, rt, t, tp)

    return _p_sample_apply

  def create_head_apply(self) -> Callable:
    model = self.model
    state = self.state
    ema_enabled = self.a.ema_enabled

    def head_apply(feat, rt, tt):
      mu = model.head.apply(
        get_params_dict(state, key="head", ema=ema_enabled), feat, rt, tt, mutable=False
      )
      return mu

    return head_apply

  def create_backbone_apply(self) -> Callable:
    model = self.model
    state = self.state
    ema_enabled = self.a.ema_enabled

    def backbone_apply(img):
      feat = model.backbone.apply(
        get_vars_dict(state, key="backbone", ema=ema_enabled), img, mutable=False
      )
      return feat

    return backbone_apply

  # --- test ---

  @abc.abstractmethod
  def create_test_dataset(self):
    # the dataset should have __iter__ and __next__ implemented
    # the dataset should return (indices, batch_data) pair
    raise NotImplementedError

  @abc.abstractmethod
  def create_test_evaluator(self, dataset):
    raise NotImplementedError

  @abc.abstractmethod
  def create_get_batch_test_fn(self):
    raise NotImplementedError

  def create_get_flat_batch_test_fn(self):
    _get_batch_test_fn = self.create_get_batch_test_fn()

    def get_flat_batch_test(seed, batch, n_slices=256):
      b = _get_batch_test_fn(seed, batch, n_slices=n_slices)
      return jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), b)

    return get_flat_batch_test

  def post_process(self, r: np.ndarray) -> np.ndarray:
    r = jnp.array(r)
    orig_shape = r.shape
    batch_shape = orig_shape[:-1]
    last_dim = orig_shape[-1]

    r = r.reshape((-1, last_dim))
    r = jax.vmap(lambda r: self.lie_metrics.as_vec(r))(r)
    r = r.reshape(tuple(batch_shape) + (r.shape[-1],))

    return np.asarray(r)

  def run_test(self, opt: Options, opt_path: str):
    self.assert_setup()
    print(" Testing")

    noise_schedule = self.noise_schedule

    timesteps = noise_schedule.timesteps
    sample_steps = opt.steps

    if sample_steps > timesteps:
      # skip
      return

    self.opt = opt
    self.opt_path = opt_path

    # set seed
    set_seed(self.a.seed)
    data_rng_seq = hk.PRNGSequence(self.a.seed + 1)
    samp_rng_seq = hk.PRNGSequence(self.a.seed + 2)
    # load dataset
    test_dataset = self.create_test_dataset()
    test_evaluator = self.create_test_evaluator(test_dataset)

    # create jitted functions
    get_flat_batch_test_fn = self.create_get_flat_batch_test_fn()
    jit_p_sample_apply = jax.jit(self.create_p_sample_apply())
    jit_backbone_apply = jax.jit(self.create_backbone_apply())

    # create time sequence
    cur_time = np.linspace(timesteps, 0, sample_steps, endpoint=False) - 1
    cur_time = cur_time.astype(np.int32).tolist()
    prev_time = cur_time[1:] + [0]
    time_seq = list(zip(cur_time, prev_time))

    pbar = tqdm(test_dataset, leave=False, desc="Sample")
    for batch_idx, batch_data in enumerate(pbar):
      # warm up jit for the first batch
      if batch_idx == 0:
        batch = get_flat_batch_test_fn(
          jax.random.PRNGKey(1), batch_data, n_slices=opt.n_slices
        )

        feat = jit_backbone_apply(batch["img"])
        jit_p_sample_apply(jax.random.PRNGKey(1), feat, batch["rt"], 1, 0)

        del batch

      batch = get_flat_batch_test_fn(
        next(data_rng_seq), batch_data, n_slices=opt.n_slices
      )

      rt = batch["rt"]
      img = batch["img"]

      start_time = time.time()
      # forward backbone
      feat = jit_backbone_apply(img)
      # forward diffusion head
      sequences = [(rt, rt)]
      for t, tp in tqdm(time_seq, leave=False, desc="Timesteps"):
        r0, rt = jit_p_sample_apply(next(samp_rng_seq), feat, rt, t, tp)
        sequences.append((r0, rt))

      inference_time = time.time() - start_time

      del batch

      bs = rt.shape[0]
      b = img.shape[0]
      s = bs // b
      t = len(sequences)

      sequence_r0, sequence_rt = zip(*sequences)
      sequence_r0 = self.post_process(sequence_r0)
      sequence_rt = self.post_process(sequence_rt)

      # (time, batch, slices, dim)
      sequence_r0 = np.array(sequence_r0).reshape((t, b, s, -1))
      sequence_rt = np.array(sequence_rt).reshape((t, b, s, -1))

      output_batch = {
        "seq_r0": sequence_r0,
        "seq_rt": sequence_rt,
        "time": inference_time,
      }

      test_evaluator.process(batch_data, output_batch)

    test_evaluator.summarize()

  # --- single-gpu train ---

  @abc.abstractmethod
  def create_train_dataset(self):
    raise NotImplementedError

  @abc.abstractmethod
  def create_get_batch_train_fn(self):
    raise NotImplementedError

  def create_get_flat_batch_train_fn(self):
    _get_batch_train_fn = self.create_get_batch_train_fn()

    def get_flat_batch_train(seed, batch, n_slices=256):
      # (b, s, ...)
      b = _get_batch_train_fn(seed, batch, n_slices=n_slices)
      # (b*s, ...)
      return jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), b)

    return get_flat_batch_train

  def create_update_fn(self) -> Callable:
    lie_metrics = self.lie_metrics
    learn_noise = self.a.learn_noise
    loss_name = self.a.loss_name

    def update_fn(state, batch):
      def loss_fn(params, batch):
        t = batch["t"].reshape((-1, 1))
        mu, new_state = state.apply_fn(
          {"params": params, "batch_stats": state.batch_stats},
          batch["img"],
          batch["rt"],
          t,
          mutable=["batch_stats"],
        )
        ta = batch["zt"] if learn_noise else batch["r0"]
        loss = jax.vmap(lie_metrics.get_distance_fn(loss_name))(ta, mu)
        return jnp.mean(loss), new_state

      (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, batch
      )
      new_state = state.apply_gradients(
        grads=grads, batch_stats=new_state["batch_stats"]
      )
      return loss, new_state

    return update_fn

  def create_update_ema_fn(self) -> Callable:
    def update_ema_fn(state, tau):
      params_ema = jax.tree_map(
        lambda p_ema, p: p_ema * tau + p * (1.0 - tau), state.params_ema, state.params
      )
      state = state.replace(params_ema=params_ema)
      return state

    return update_ema_fn

  def run_train_single(self):
    self.assert_setup()
    print(" Single-GPU training")

    # set seed
    set_seed(self.a.seed)
    rng_seq = hk.PRNGSequence(self.a.seed)
    # load dataset
    train_dataset = self.create_train_dataset()
    train_iter = iter(train_dataset)
    # create tensorboard
    summary_writer = tensorboard.SummaryWriter(self.exp_path)
    # create jitted functions
    get_flat_batch_train_fn = self.create_get_flat_batch_train_fn()
    jit_update_fn = jax.jit(self.create_update_fn())
    jit_update_ema_fn = jax.jit(self.create_update_ema_fn())

    ckpt_dir = os.path.join(self.exp_path, "ckpt")
    pbar = tqdm(range(self.a.train_steps), leave=False, desc="Train")

    for step in pbar:
      # get batch (b*s, ...)
      batch = get_flat_batch_train_fn(
        next(rng_seq), next(train_iter), n_slices=self.a.n_slices
      )
      # train one step
      loss, self.state = jit_update_fn(self.state, batch)
      if (self.a.ema_enabled) and (step % self.a.ema_steps == 0):
        self.state = jit_update_ema_fn(self.state, self.a.ema_tau)
      loss = np.array(loss).item()
      global_step = self.state.step

      if step % 5 == 0:
        pbar.set_postfix({"loss": f"{loss:.6f}"})
        summary_writer.scalar("train_loss", loss, global_step)
        summary_writer.scalar("lr", self.lr_schedule(global_step), global_step)

      if (step + 1) % self.a.ckpt_steps == 0:
        save_checkpoint(ckpt_dir=ckpt_dir, target=self.state, step=global_step)

      if jnp.isnan(loss):
        raise RuntimeError("Loss became NaN")

    save_checkpoint(ckpt_dir=ckpt_dir, target=self.state, step=global_step + 1)

    summary_writer.flush()
    return self

  # --- multi-gpu train ---

  def create_get_flat_batch_train_fn_pmap(self) -> Callable:
    _get_batch_train_fn = self.create_get_batch_train_fn()
    num_devices = self.num_devices

    def get_flat_batch_train_pmap(seed, batch, n_slices=256):
      # (b, s, ...)
      b = _get_batch_train_fn(seed, batch, n_slices=n_slices)
      # (gpu, b//gpu*s, ...)
      return jax.tree_util.tree_map(
        lambda x: x.reshape((num_devices, -1, *x.shape[2:])), b
      )

    return get_flat_batch_train_pmap

  def create_update_fn_pmap(self, axis_name="gpu") -> Callable:
    lie_metrics = self.lie_metrics
    learn_noise = self.a.learn_noise
    loss_name = self.a.loss_name

    def update_fn(state, batch):
      def loss_fn(params, batch):
        t = batch["t"].reshape((-1, 1))
        mu, new_state = state.apply_fn(
          {"params": params, "batch_stats": state.batch_stats},
          batch["img"],
          batch["rt"],
          t,
          mutable=["batch_stats"],
        )
        ta = batch["zt"] if learn_noise else batch["r0"]
        loss = jax.vmap(lie_metrics.get_distance_fn(loss_name))(ta, mu)
        return jnp.mean(loss), new_state

      (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, batch
      )
      loss = jax.lax.pmean(loss, axis_name=axis_name)
      grads = jax.lax.pmean(grads, axis_name=axis_name)
      new_state = state.apply_gradients(
        grads=grads, batch_stats=new_state["batch_stats"]
      )
      return loss, new_state

    return jax.pmap(update_fn, axis_name=axis_name, devices=self.devices)

  def create_sync_batch_stats_fn_pmap(self):
    _cross_replica_mean = jax.pmap(
      lambda x: jax.lax.pmean(x, axis_name="x"), axis_name="x", devices=self.devices
    )

    def sync_batch_stats(state):
      return state.replace(batch_stats=_cross_replica_mean(state.batch_stats))

    return sync_batch_stats

  def create_update_ema_fn_pmap(self, axis_name="gpu") -> Callable:
    def update_ema_fn(state, tau):
      params_ema = jax.tree_map(
        lambda p_ema, p: p_ema * tau + p * (1.0 - tau), state.params_ema, state.params
      )
      state = state.replace(params_ema=params_ema)
      return state

    return jax.pmap(
      update_ema_fn, in_axes=(0, None), axis_name=axis_name, devices=self.devices
    )

  def run_train_parallel(self):
    self.assert_setup()
    print(" Multi-GPU training enabled:")
    print(f"   Number of devices: {self.num_devices}")
    print(f"   Device list: {self.devices}")

    # set seed
    set_seed(self.a.seed)
    rng_seq = hk.PRNGSequence(self.a.seed)
    # load dataset
    assert (
      self.a.batch_size % self.num_devices == 0
    ), f"Batch size '{self.a.batch_size}' must be dividable by number of devices '{self.num_devices}'"
    train_dataset = self.create_train_dataset()
    train_iter = iter(train_dataset)
    # create tensorboard
    summary_writer = tensorboard.SummaryWriter(self.exp_path)
    # create jitted functions
    p_get_flat_batch_train_fn = self.create_get_flat_batch_train_fn_pmap()
    p_update_fn = self.create_update_fn_pmap()
    p_sync_batch_stats = self.create_sync_batch_stats_fn_pmap()
    p_update_ema_fn = self.create_update_ema_fn_pmap()

    # create replica on each device
    self.state = jax_utils.replicate(self.state, devices=self.devices)

    ckpt_dir = os.path.join(self.exp_path, "ckpt")
    pbar = tqdm(range(self.a.train_steps), leave=False, desc="Train")

    for step in pbar:
      # get batch (gpu, b*s, ...)
      batch = p_get_flat_batch_train_fn(
        next(rng_seq), next(train_iter), n_slices=self.a.n_slices
      )
      # train one step
      loss, state = p_update_fn(self.state, batch)
      # sync batch norm
      self.state = p_sync_batch_stats(state)
      # update ema
      if (self.a.ema_enabled) and (step % self.a.ema_steps == 0):
        self.state = p_update_ema_fn(self.state, self.a.ema_tau)
      loss = np.array(jax_utils.unreplicate(loss)).item()
      global_step = jax_utils.unreplicate(self.state.step)

      if step % 5 == 0:
        pbar.set_postfix({"loss": f"{loss:.6f}"})
        summary_writer.scalar("train_loss", loss, global_step)
        summary_writer.scalar("lr", self.lr_schedule(global_step), global_step)

      if (step + 1) % self.a.ckpt_steps == 0:
        state = jax_utils.unreplicate(self.state)
        save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=global_step)

      if jnp.isnan(loss):
        raise RuntimeError("Loss became NaN")

    self.state = jax_utils.unreplicate(self.state)
    save_checkpoint(ckpt_dir=ckpt_dir, target=self.state, step=global_step + 1)

    summary_writer.flush()
    return self

  def run_train(self):
    if self.num_devices > 1:
      return self.run_train_parallel()
    else:
      return self.run_train_single()
