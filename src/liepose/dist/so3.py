import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxlie import SO3

from .base import LieDist


def _isotropic_gaussian_so3_small(omg, scale):
  """Borrowed from: https://github.com/tomato1mule/edf/blob/1dd342e849fcb34d3eb4b6ad2245819abbd6c812/edf/dist.py#L99
  This function implements the approximation of the density function of omega of the isotropic Gaussian distribution.
  """
  eps = scale**2
  # TODO: check for stability and maybe replace by limit in 0 for small values
  small_number = 1e-9
  small_num = small_number / 2
  small_dnm = (
    1 - jnp.exp(-1.0 * jnp.pi**2 / eps) * (2 - 4 * (np.pi**2) / eps)
  ) * small_number

  return (
    0.5
    * jnp.sqrt(jnp.pi)
    * (eps**-1.5)
    * jnp.exp((eps - (omg**2 / eps)) / 4)
    / (jnp.sin(omg / 2) + small_num)
    * (
      small_dnm
      + omg
      - (
        (omg - 2 * jnp.pi) * jnp.exp(jnp.pi * (omg - jnp.pi) / eps)
        + (omg + 2 * jnp.pi) * jnp.exp(-jnp.pi * (omg + jnp.pi) / eps)
      )
    )
  )


def _isotropic_gaussian_so3(omg, scale, lmax=None):
  """Borrowed from: https://github.com/tomato1mule/edf/blob/1dd342e849fcb34d3eb4b6ad2245819abbd6c812/edf/dist.py#L82
  This function implements the density function of omega of the isotropic Gaussian distribution.
  """
  eps = scale**2

  if lmax is None:
    lmax = max(int(3.0 / np.sqrt(eps)), 2)

  small_number = 1e-9
  sum = 0.0
  # TODO: replace by a scan
  for l in range(lmax + 1):
    sum = sum + (2 * l + 1) * jnp.exp(-l * (l + 1) * eps) * (
      jnp.sin((l + 0.5) * omg) + (l + 0.5) * small_number
    ) / (jnp.sin(omg / 2) + 0.5 * small_number)
  return sum


def isotropic_gaussian_so3(omg, scale, force_small=False):
  if force_small:
    return _isotropic_gaussian_so3_small(omg, scale / jnp.sqrt(2))
  else:
    return jax.lax.cond(
      scale < 1,
      lambda x: _isotropic_gaussian_so3_small(x, scale / jnp.sqrt(2)),
      lambda x: _isotropic_gaussian_so3(x, scale / jnp.sqrt(2), lmax=3),
      omg,
    )


class NormalSO3_IG(LieDist):
  _min_num_samples = 1024
  _min_scale = 1e-4

  def __init__(
    self,
    mean: SO3,
    scale: float,
    approx: bool = True,
    num_samples: int = 1024,
    power: float = 3.0,
  ):
    assert isinstance(mean, SO3)
    self._mean = mean
    self._scale = jnp.maximum(scale, self._min_scale)
    self._approx = approx
    self._num_samples = int(max(num_samples, self._min_num_samples))
    self._power = max(power, 1.0)

    self._generate_cdf()

  def _generate_cdf(self):
    x = jnp.linspace(0.0, 1.0, self._num_samples)
    # numerically stable
    self._x = (x**self._power) * jnp.pi
    y = (1 - jnp.cos(self._x)) / jnp.pi * self._f(self._x)
    y = jnp.cumsum(y) * jnp.pi / self._num_samples
    self._y = y / y.max()

  @property
  def mean(self) -> jnp.array:
    return self._mean

  @property
  def scale(self) -> jnp.array:
    return self._scale

  @property
  def dtype(self) -> jnp.array:
    return self._mean.dtype

  def _f(self, x: jnp.array) -> jnp.array:
    return isotropic_gaussian_so3(x, self.scale, force_small=self._approx)

  def prob(self, x: SO3) -> jnp.array:
    dx = (self.mean.inverse() @ x).log()
    ang = jnp.linalg.norm(dx, axis=-1)
    return self._f(ang)

  def log_prob(self, x: SO3) -> jnp.array:
    return jnp.log(self.prob(x) + 1e-9)

  def mode(self) -> SO3:
    return self.mean

  def _sample(self, seed, n) -> jnp.array:
    """Sample n tangent vectors"""
    key1, key2 = jax.random.split(seed, 2)
    rnd = jax.random.uniform(shape=n, key=key1)
    ang = jnp.interp(rnd, self._y, self._x)
    axis = jax.random.normal(shape=n + (3,), key=key2)
    axis = axis / jnp.linalg.norm(axis, axis=-1, keepdims=True)
    tan = ang[..., jnp.newaxis] * axis
    return tan

  def sample(self, seed=None, n=[]) -> SO3:
    """Sample n rotations"""
    n = tuple(n) if hasattr(n, "__iter__") else (n,)
    tan = self._sample(seed, n)

    shape = tan.shape[:-1]
    tan = tan.reshape(-1, 3)
    quat = jax.vmap(lambda tan: (self.mean @ SO3.exp(tan)).wxyz)(tan)

    so3 = SO3(quat.reshape(shape + (4,)))
    return so3

  @classmethod
  def _sample_unit(cls, seed, n) -> jnp.array:
    return cls(SO3.identity(), 1.0, approx=True)._sample(seed, n)


class NormalSO3_Flat(LieDist):
  """Concentrated Gaussian SO3"""

  _min_scale = 1e-4

  def __init__(self, mean: SO3, scale: float, **kwargs):
    assert isinstance(mean, SO3)
    self._mean = mean
    self._scale = jnp.maximum(scale, self._min_scale)

  @property
  def mean(self) -> jnp.array:
    return self._mean

  @property
  def scale(self) -> jnp.array:
    return self._scale

  @property
  def dtype(self) -> jnp.array:
    return self._mean.dtype

  def prob(self, x: SO3) -> jnp.ndarray:
    logp = self.log_prob(x)
    return jnp.exp(logp)

  def log_prob(self, x: SO3) -> jnp.array:
    var = self._scale**2
    log_scale = jnp.log(self._scale)
    z = math.log(math.sqrt(2 * math.pi))
    dff = (self.mean.inverse() @ x).log()
    return -((dff**2) / (2 * var) - log_scale - z).sum()

  def mode(self) -> SO3:
    return self.mean

  def _sample(self, seed, n) -> jnp.array:
    tan = jax.random.normal(shape=n + (3,), key=seed)
    tan = tan * self._scale
    return SO3.exp(tan).log()

  def sample(self, seed=None, n=[]) -> SO3:
    """Sample n rotations"""
    n = tuple(n) if hasattr(n, "__iter__") else (n,)
    tan = self._sample(seed, n)

    shape = tan.shape[:-1]
    tan = tan.reshape(-1, 3)
    quat = jax.vmap(lambda tan: (self.mean @ SO3.exp(tan)).wxyz)(tan)

    so3 = SO3(quat.reshape(shape + (4,)))
    return so3

  @classmethod
  def _sample_unit(cls, seed, n) -> jnp.array:
    return cls(SO3.identity(), 1.0, approx=True)._sample(seed, n)


# NormalSO3 = NormalSO3_IG
NormalSO3 = NormalSO3_Flat
