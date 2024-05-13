import abc

from jaxlie import MatrixLieGroup


class LieDist(abc.ABC):
  @abc.abstractmethod
  def prob(self, x: MatrixLieGroup) -> MatrixLieGroup:
    """Probability"""

  @abc.abstractmethod
  def log_prob(self, x: MatrixLieGroup) -> MatrixLieGroup:
    """Log Probability"""

  @abc.abstractmethod
  def sample(self, seed, n=()) -> MatrixLieGroup:
    """Sample lie object"""
