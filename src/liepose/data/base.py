import abc


class BaseEvaluator(abc.ABC):
  @abc.abstractmethod
  def reset(self):
    """Reset evaluator"""
    pass

  @abc.abstractmethod
  def process(self, inputs, outputs):
    """Process batched inputs and outputs"""
    pass

  @abc.abstractmethod
  def summarize(self):
    """Calculate evaluation metrics"""
    pass
