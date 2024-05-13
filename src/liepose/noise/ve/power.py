import numpy as np

from .base import BaseVENoiseSchedule


class PowerNoiseSchedule(BaseVENoiseSchedule):
  def __init__(
    self,
    alpha_start: float = 1e-8,
    alpha_end: float = 1.0,
    timesteps: int = 500,
    power: float = 3.0,
  ):
    self.alpha_start = alpha_start
    self.alpha_end = alpha_end
    self.power = power

    super().__init__(timesteps)

  def create_alphas(self, timesteps):
    return (
      np.linspace(
        self.alpha_start ** (1 / self.power),
        self.alpha_end ** (1 / self.power),
        timesteps,
      )
      ** self.power
    )
