import numpy as np

from .base import BaseVPNoiseSchedule


class PowerNoiseSchedule(BaseVPNoiseSchedule):
  def __init__(
    self,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    timesteps: int = 500,
    power: float = 1.0,
  ):
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.timesteps = timesteps
    self.power = power

    super().__init__(timesteps)

  def create_betas(self, timesteps):
    return (
      np.linspace(
        self.beta_start ** (1 / self.power),
        self.beta_end ** (1 / self.power),
        timesteps,
        dtype=np.float64,
      )
      ** self.power
    )
