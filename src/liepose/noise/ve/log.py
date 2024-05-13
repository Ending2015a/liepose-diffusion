import numpy as np

from .base import BaseVENoiseSchedule


class LogNoiseSchedule(BaseVENoiseSchedule):
  def __init__(
    self, alpha_start: float = 1e-8, alpha_end: float = 1.0, timesteps: int = 500
  ):
    self.alpha_start = alpha_start
    self.alpha_end = alpha_end

    super().__init__(timesteps)

  def create_alphas(self, timesteps):
    return np.exp(
      np.linspace(np.log(self.alpha_start), np.log(self.alpha_end), timesteps)
    )
