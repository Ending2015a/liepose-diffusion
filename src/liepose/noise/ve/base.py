import jax.numpy as jnp
import numpy as np

from ..base import BaseNoiseSchedule


class BaseVENoiseSchedule(BaseNoiseSchedule):
  def __init__(self, timesteps: int = 500):
    self.timesteps = timesteps

    alphas = self.create_alphas(timesteps + 1)
    betas = np.diff(alphas, prepend=0)
    alphas_prev = np.append(0.0, alphas[:-1])

    self.betas = jnp.array(betas, dtype=jnp.float32)
    self.alphas = jnp.array(alphas, dtype=jnp.float32)
    self.alphas_prev = jnp.array(alphas_prev, dtype=jnp.float32)

    self.sqrt_betas = jnp.array(np.sqrt(betas), dtype=jnp.float32)
    self.sqrt_alphas = jnp.array(np.sqrt(alphas), dtype=jnp.float32)
    self.sqrt_alphas_prev = jnp.array(np.sqrt(alphas_prev), dtype=jnp.float32)

    self.coef1 = self.sqrt_betas / self.sqrt_alphas * self.sqrt_betas
    self.coef2 = self.sqrt_alphas_prev / self.sqrt_alphas * self.sqrt_betas

  def create_alphas(self, timesteps):
    return np.linspace(0.01, 1.0, timesteps)
