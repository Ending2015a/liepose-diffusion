import jax
import jax.numpy as jnp
from jaxlie import R3SO3

from .base import LieDist
from .so3 import NormalSO3_IG, NormalSO3_Flat


class _NormalR3SO3(LieDist):
  SO3_Dist: LieDist
  _min_scale = 1e-4

  def __init__(self, mean: R3SO3, scale: float, **kwargs):
    assert isinstance(mean, R3SO3)
    self._mean = mean
    self._scale = jnp.maximum(scale, self._min_scale)

    self._so3_dist = self.SO3_Dist(self._mean.rotation(), self._scale, **kwargs)

  @property
  def mean(self) -> R3SO3:
    return self._mean

  @property
  def scale(self) -> float:
    return self._scale

  @property
  def dtype(self):
    return self._mean.dtype

  def prob(self, x: R3SO3) -> jnp.array:
    return jnp.exp(self.log_prob(x))

  def log_prob(self, x: R3SO3) -> jnp.array:
    dx = (self.mean.inverse @ x).log()
    dx_tran = dx[:3]
    dx_rot = dx[3:]
    # calculate rotation log prob
    ang = jnp.linalg.norm(dx_rot, axis=-1)
    logp_rot = jnp.log(self._so3_dist._f(ang) + 1e-9)
    # calculate translation log prob
    scale = self.scale
    var = scale**2
    log_scale = jnp.log(scale)
    logp_tran = (
      -(dx_tran**2) / (2 * var) - log_scale - jnp.log(jnp.sqrt(2 * jnp.pi))
    ).sum(dim=-1)
    logp = logp_rot + logp_tran
    return logp

  def mode(self) -> R3SO3:
    return self.mean

  def _sample(self, seed, n) -> jnp.array:
    key1, key2 = jax.random.split(seed, 2)
    # sample rotation
    dx_rot = self._so3_dist._sample(key1, n)
    dx_tran = jax.random.normal(shape=n + (3,), key=key2) * self.scale
    dx = jnp.concatenate([dx_tran, dx_rot], axis=-1)
    return dx

  def sample(self, seed=None, n=[]) -> R3SO3:
    n = tuple(n) if hasattr(n, "__iter__") else (n,)
    tan = self._sample(seed, n)

    shape = tan.shape[:-1]
    tan = tan.reshape(-1, 6)
    quat_t = jax.vmap(lambda tan: (self.mean @ R3SO3.exp(tan)).wxyz_xyz)(tan)

    se3 = R3SO3(quat_t.reshape(shape + (7,)))
    return se3

  @classmethod
  def _sample_unit(cls, seed, n) -> jnp.array:
    return cls(R3SO3.identity(), 1.0, approx=True)._sample(seed, n)


class NormalR3SO3_IG(_NormalR3SO3):
  SO3_Dist = NormalSO3_IG


class NormalR3SO3_Flat(_NormalR3SO3):
  SO3_Dist = NormalSO3_Flat


NormalR3SO3 = NormalR3SO3_Flat
