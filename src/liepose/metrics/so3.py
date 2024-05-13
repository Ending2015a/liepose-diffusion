from typing import Callable

import jax.numpy as jnp
from jaxlie import SO3

from ..utils import is_jaxndarray


def is_mat(t):
  return is_jaxndarray(t, (3, 3))


def is_mat44(t):
  return is_jaxndarray(t, (4, 4))


def is_tan(t):
  return is_jaxndarray(t, (3,)) and not is_mat(t)


def is_vec(t):
  return is_jaxndarray(t, (4,)) and not is_mat44(t)


def is_vec6d(t):
  return is_jaxndarray(t, (6,))


def is_vec9d(t):
  return is_jaxndarray(t, (9,))


def as_lie(t) -> SO3:
  if isinstance(t, SO3):
    return t
  if is_mat(t):
    # from 3x3 matrix
    return SO3.from_matrix(t)
  if is_mat44(t):
    # from 4x4 matrix
    return SO3.from_matrix(as_mat(t))
  if is_tan(t):
    # from tangent vector
    return SO3.exp(t)
  if is_vec(t):
    # from quaterion
    return SO3(t)
  if is_vec6d(t):
    # from vec6d
    return SO3.from_matrix(as_mat(t))
  if is_vec9d(t):
    # from vec9d
    return SO3.from_matrix(as_mat(t))
  raise ValueError(t.shape)


def as_mat(t) -> jnp.ndarray:
  "Return 3x3 matrix"
  if is_mat(t):
    return t
  if is_mat44(t):
    # from 4x4 matrix
    return t[..., :3, :3]
  if is_vec6d(t):
    return SO3.orthogonalize(t)
  if is_vec9d(t):
    return SO3.orthogonalize(t.reshape((3, 3)).swapaxes(-2, -1))
  return as_lie(t).as_matrix()


def as_vec(t) -> jnp.ndarray:
  if is_vec(t):
    return t
  return as_lie(t).wxyz


def as_vec6d(t) -> jnp.ndarray:
  if is_vec6d(t):
    return t
  if is_vec9d(t):
    return t[..., :6]
  if not is_mat(t):
    t = as_mat(t)
  return flatten_last2(t.swapaxes(-2, -1))[..., :6]


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
  elif repr == "vec6d":
    return as_vec6d(t)
  raise ValueError(repr)


def get_repr_size(repr: str) -> int:
  if repr == "lie":
    return None
  elif repr == "mat":
    return (3, 3)
  elif repr == "tan":
    return 3
  elif repr == "vec":
    return 4
  elif repr == "vec6d":
    return 6
  raise ValueError(repr)


def flatten_last2(t):
  if len(t.shape) < 2:
    return t
  return t.reshape((*t.shape[:-2], t.shape[-2] * t.shape[-1]))


def geodesic_distance(x1, x2):
  """Angular geodesic distance"""
  x1 = as_lie(x1)
  x2 = as_lie(x2)
  dff = x2 @ x1.inverse()
  return (dff.log() ** 2).sum(axis=-1)


def chordal_distance(x1, x2):
  m1 = as_mat(x1)
  m2 = as_mat(x2)
  m = (m1 - m2) ** 2
  m = flatten_last2(m)
  return m.sum(axis=-1)


def quaternion_distance(x1, x2):
  q1 = as_vec(x1)
  q2 = as_vec(x2)
  p = ((q1 + q2) ** 2).sum(axis=-1)
  m = ((q1 - q2) ** 2).sum(axis=-1)
  return jnp.minimum(p, m)


def hyperbolic_distance(x1, x2):
  x1 = as_tan(x1)
  x2 = as_tan(x2)
  return ((x1 - x2) ** 2).sum(axis=-1)


def chordal6d_distance(x1, x2):
  """6D rotation representation"""
  m1 = as_vec6d(x1)
  m2 = as_vec6d(x2)
  m = (m1 - m2) ** 2
  return (m).sum(axis=-1)


# =========

distance_dict = {
  "geodesic": geodesic_distance,
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
