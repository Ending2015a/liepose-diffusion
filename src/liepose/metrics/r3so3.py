import functools
from typing import Callable

import jax.numpy as jnp
from jaxlie import R3SO3

from ..utils import is_jaxndarray
from .so3 import distance_dict as so3_distance_dict


def is_mat(t):
  return is_jaxndarray(t, (4, 4))


def is_tan(t):
  return is_jaxndarray(t, (6,))


def is_vec(t):
  return is_jaxndarray(t, (7,))


def is_vec9d(t):
  return is_jaxndarray(t, (9,))


def as_lie(t) -> R3SO3:
  if isinstance(t, R3SO3):
    return t
  if is_mat(t):
    # 4x4 matrix
    return R3SO3.from_matrix(t)
  if is_tan(t):
    # 3d rotation + 3d translation
    return R3SO3.exp(t)
  if is_vec(t):
    # 4d rotation + 3d translation
    return R3SO3(t)
  if is_vec9d(t):
    # 6d rotation + 3d translation
    return R3SO3.from_matrix(as_mat(t))
  raise ValueError(t.shape)


def as_mat(t) -> jnp.ndarray:
  """Return 4x4 matrix"""
  if is_mat(t):
    return t
  if is_vec9d(t):
    # 6d rotation + 3d translation
    return R3SO3.orthogonalize(t)
  return as_lie(t).as_matrix()


def as_vec(t) -> jnp.ndarray:
  if is_vec(t):
    return t
  return as_lie(t).wxyz_xyz


def as_vec9d(t) -> jnp.ndarray:
  if is_vec9d(t):
    return t
  if not is_mat(t):
    t = as_mat(t)
  rot = flatten_last2(t[..., :3, :3].swapaxes(-2, -1))[..., :6]
  tran = t[..., :3, 3]
  return jnp.concatenate([rot, tran], axis=-1)


def as_tan(t) -> jnp.ndarray:
  """Return tangent vector"""
  if is_tan(t):
    return t
  return as_lie(t).log()


def as_repr(t, repr: str) -> jnp.ndarray:
  if repr == "lie":
    return as_lie(t)
  elif repr == "mat":
    return as_mat(t)
  elif repr == "tan":
    return as_tan(t)
  elif repr == "vec":
    return as_vec(t)
  elif repr == "vec9d":
    return as_vec9d(t)
  raise ValueError(repr)


def get_repr_size(repr: type) -> int:
  if repr == "lie":
    return None
  elif repr == "mat":
    return (4, 4)
  elif repr == "tan":
    return 6
  elif repr == "vec":
    return 7
  elif repr == "vec9d":
    return 9
  raise ValueError(repr)


def flatten_last2(t):
  if len(t.shape) < 2:
    return t
  return t.reshape((*t.shape[:-2], t.shape[-2] * t.shape[-1]))


def euclidean_distance(x1, x2):
  return ((x1 - x2) ** 2).sum(axis=-1)


def chordal_distance(x1, x2, w=1.0):
  m1 = as_mat(x1)
  m2 = as_mat(x2)
  m = (m1 - m2) ** 2
  m = flatten_last2(m)
  return m.sum(axis=-1)


def chordal6d_distance(x1, x2, w=1.0):
  m1 = as_vec9d(x1)
  m2 = as_vec9d(x2)
  m = (m1 - m2) ** 2
  return m[..., :6].sum(axis=-1) + w * m[..., 6:].sum(axis=-1)


def hyperbolic_distance(x1, x2, w=1.0):
  x1 = as_tan(x1)
  x2 = as_tan(x2)
  x = (x1 - x2) ** 2
  return x[..., 3:].sum(axis=-1) + w * x[..., :3].sum(axis=-1)


def _factory(distance_fn):
  def _so3_trans_distance(x1, x2, distance_fn, w=1.0):
    x1 = as_lie(x1)
    x2 = as_lie(x2)
    r1 = x1.rotation()
    r2 = x2.rotation()
    t1 = x1.translation()
    t2 = x2.translation()
    dist = distance_fn(r1, r2)
    dist += w * euclidean_distance(t1, t2)
    return dist

  return functools.partial(_so3_trans_distance, distance_fn=distance_fn)


geodesic_distance = _factory(so3_distance_dict["geodesic"])
quaternion_distance = _factory(so3_distance_dict["quaternion"])

# =======
distance_dict = {
  "geodesic": geodesic_distance,  # geo(R) + dist(T)
  "chordal": chordal_distance,
  "quaternion": quaternion_distance,
  "hyperbolic": hyperbolic_distance,
  "chordal6d": chordal6d_distance,
}


def get_distance_fn(type="quaternion") -> Callable:
  type = str(type).lower()
  assert type in distance_dict.keys()

  distance_fn = distance_dict[type]
  return distance_fn
