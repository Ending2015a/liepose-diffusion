import jax.numpy as jnp
import numpy as np

from ..base import BaseNoiseSchedule


class BaseVPNoiseSchedule(BaseNoiseSchedule):
  def __init__(self, timesteps: int = 500):
    self.timesteps = timesteps

    betas = self.create_betas(timesteps)

    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas, axis=0)
    alphas_bar_prev = np.append(1.0, alphas_bar[:-1])

    self.betas = jnp.array(betas, dtype=jnp.float32)
    self.alphas = jnp.array(alphas, dtype=jnp.float32)
    self.alphas_bar = jnp.array(alphas_bar, dtype=jnp.float32)
    self.alphas_bar_prev = jnp.array(alphas_bar_prev, dtype=jnp.float32)

    self.sqrt_betas = jnp.array(np.sqrt(betas), dtype=jnp.float32)
    self.sqrt_alphas_bar = jnp.array(np.sqrt(alphas_bar), dtype=jnp.float32)
    self.sqrt_alphas_bar_prev = jnp.array(np.sqrt(alphas_bar_prev), dtype=jnp.float32)

    self.sqrt_one_minus_alphas_bar = jnp.array(
      np.sqrt(1 - self.alphas_bar), dtype=jnp.float32
    )

    self.sqrt_one_minus_alphas_bar_prev = jnp.array(
      np.sqrt(1 - self.alphas_bar_prev), dtype=jnp.float32
    )

  def create_betas(self, timesteps):
    return np.linspace(1e-4, 0.02, timesteps)
