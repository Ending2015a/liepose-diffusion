from typing import Union

import jax
from jaxlie import MatrixLieGroup
from jaxlie.hints import Array, Scalar

Element = Union[Array, Scalar, MatrixLieGroup]

__all__ = ["scale", "add", "sub", "lsub", "rsub"]

"""LieGroup operations
lsub: Left-subtraction, e.g.
  x = lsub(add(x, y), y) ====> (XY) * Y^(-1) = X
  -x = lsub(y, add(x, y)) ====> Y * (XY)^(-1) = Y * Y^(-1) * X^(-1) = X^(-1)
rsub: Right-subtraction, e.g.
  y = rsub(x, add(x, y)) ====> X^(-1) * (XY) = Y
  -y = rsub(add(x, y), x) ====> (XY)^(-1) * X = Y^(-1) * X^(-1) * X = Y^(-1)
"""


def _scale(c, x):
  return type(x).exp(c * x.log())


def _add(y, x):
  return y @ x


def _lsub(y, x):
  return y @ x.inverse()


def _rsub(y, x):
  return y.inverse() @ x


def scale(c: Scalar, x: Element) -> Element:
  if isinstance(x, MatrixLieGroup):
    return _scale(c, x)
  else:
    return c * x


def add(y: Element, x: Element) -> Element:
  if isinstance(x, MatrixLieGroup):
    return _add(y, x)
  else:
    return y + x


def lsub(y: Element, x: Element) -> Element:
  if isinstance(x, MatrixLieGroup):
    return _lsub(y, x)
  else:
    return y - x


def rsub(y: Element, x: Element) -> Element:
  if isinstance(x, MatrixLieGroup):
    return _rsub(y, x)
  else:
    return -y + x


# default use left-sub
sub = lsub


if __name__ == "__main__":
  from jaxlie import SE3

  x = SE3.sample_uniform(jax.random.PRNGKey(1))
  y = SE3.sample_uniform(jax.random.PRNGKey(2))

  print("x:", x.log())
  print("y:", y.log())
  print("(x+y)-y=x : ", lsub(add(x, y), y).log())
  print("y-(x+y)=-x : ", lsub(y, add(x, y)).log())
  print("-x+(x+y)=y : ", rsub(x, add(x, y)).log())
  print("-(x+y)+x=-y : ", rsub(add(x, y), x).log())
