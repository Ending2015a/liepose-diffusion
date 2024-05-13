import os
import subprocess

import jax
import jax.numpy as jnp
import numpy as np


def is_jaxarray(t):
  return isinstance(t, jnp.ndarray)


def is_jaxndarray(t, nd=None):
  if not isinstance(t, jnp.ndarray):
    return False
  if nd is not None:
    if not isinstance(nd, (tuple, list)):
      nd = [nd]
    nd = tuple(nd)
    return t.shape[-len(nd) :] == nd
  return True


def get_workdir(default=None):
  return os.environ.get("WORKDIR", default)


def get_datadir(default=None):
  return os.environ.get("DATADIR", default)


def workdir(path, default="./"):
  return os.path.join(get_workdir(default), path)


def datadir(path, default="/"):
  return os.path.join(get_datadir(default), path)


def save_params(path, params):
  leaves, treedef = jax.tree_util.tree_flatten(params)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  np.savez(path, *leaves, treedef=treedef)


def load_params(path, allow_pickle=True):
  npfile = np.load(path, allow_pickle=allow_pickle)
  if "treedef" not in npfile.files:
    raise ValueError(f"Numpy file dose not contain 'treedef' key: {path}")
  treedef = npfile["treedef"].item()
  npfile.files.remove("treedef")
  leaves = [npfile[arr] for arr in npfile.files]
  params = jax.tree_util.tree_unflatten(treedef, leaves)
  return params


def encode_video(
  path,
  save_path,
  in_fps=20,
  out_fps=20,
):
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  cmdline = (
    "ffmpeg",
    "-nostats",  # disable encoding process info
    "-loglevel",
    "error",  # suppress warnings
    "-y",  # replace existing file
    # input stream settings
    "-framerate",
    f"{in_fps:d}",
    "-i",
    f"{path}",  # input path
    # output stream settings
    "-c:v",
    "libx264",
    "-vf",
    f"fps={out_fps:d}",
    "-pix_fmt",
    "yuv420p",
    f"{save_path}",
  )
  subprocess.call(cmdline)


def try_restore_state(ckpt_dir, state, step=None):
  from flax.training import checkpoints

  if step is not None:
    ckpt_path = checkpoints._checkpoint_path(ckpt_dir, step)
    if not os.path.exists(ckpt_path):
      # failed to restore state
      return None, ckpt_path
  else:
    if not os.path.exists(ckpt_dir):
      # No checkpoints were found
      return None, None
    if os.path.isdir(ckpt_dir):
      ckpt_path = checkpoints.latest_checkpoint(ckpt_dir)
      if not ckpt_path:
        # No checkpoints were found
        return None, None
    else:
      ckpt_path = ckpt_dir

  restored_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=state)

  if restored_state is state:
    return None, ckpt_path
  return restored_state, ckpt_path


def ntfy_log(msg: str):
  key = os.environ.get("NTFY_ACCESS_KEY", None)
  if not key:
    return

  import requests

  requests.post(f"https://ntfy.sh/{key}", data=msg.encode(encoding="utf-8"))


class AttrDict(dict):
  def __new__(cls, *args, **kwargs):
    self = super().__new__(cls, *args, **kwargs)
    self.__dict__ = self
    return self

  def __repr__(self):
    return f"{type(self).__name__}({super().__repr__()})"
