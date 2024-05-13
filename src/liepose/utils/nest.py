import itertools
from typing import Any, Callable, Iterator, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np


def iter_nested(data: Any, sortkey: bool = False) -> Iterator[Any]:
  """Iterate over nested data structure
  Note: Use `tuple` instead of `list`. A list type
  object is treated as an item.
  For example:
  >>> data = {'a': (1, 2), 'b': 3}
  >>> list(v for v in iter_nested(data))
  [1, 2, 3]
  Args:
    data (Any): A nested data.
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """

  def _inner_iter_nested(data):
    if isinstance(data, dict):
      keys = sorted(data.keys()) if sortkey else data.keys()
      for k in keys:
        yield from _inner_iter_nested(data[k])
    elif isinstance(data, tuple):
      for v in data:
        yield from _inner_iter_nested(v)
    else:
      yield data

  return _inner_iter_nested(data)


def map_nested(data: Any, op: Callable, *args, sortkey: bool = False, **kwargs) -> Any:
  """A nested version of map function
  NOTE: Use `tuple` instead of `list`. A list type
  object is treated as an item.
  Args:
    data (Any): A nested data
    op (Callable): A function operate on each data
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """
  if not callable(op):
    raise ValueError("`op` must be a callable")

  def _inner_map_nested(data):
    if isinstance(data, dict):
      keys = sorted(data.keys()) if sortkey else data.keys()
      return {k: _inner_map_nested(data[k]) for k in keys}
    elif isinstance(data, tuple):
      return tuple(_inner_map_nested(v) for v in data)
    else:
      return op(data, *args, **kwargs)

  return _inner_map_nested(data)


def iter_nested_tuple(
  data_tuple: Tuple[Any], sortkey: bool = False
) -> Iterator[Tuple[Any]]:
  """Iterate over a tuple of nested structures. Similar to iter_nested
  but it iterates each of each nested data in the input tuple.
  For example:
  >>> a = {'x': 1, 'y': (2, 3)}
  >>> b = {'u': 4, 'v': (5, 6)}
  >>> list(iter_nested_tuple((a, b)))
  [(1, 4), (2, 5), (3, 6)]
  Args:
    data_tuple (Tuple[Any]): A tuple of nested data.
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """
  if not isinstance(data_tuple, tuple):
    raise TypeError("`data_tuple` only accepts tuple, " f"got {type(data_tuple)}")

  def _inner_iter_nested(data_tuple):
    if isinstance(data_tuple[0], dict):
      keys = data_tuple[0].keys()
      keys = sorted(keys) if sortkey else keys
      for k in keys:
        yield from _inner_iter_nested(tuple(data[k] for data in data_tuple))
    elif isinstance(data_tuple[0], tuple):
      for k in range(len(data_tuple[0])):
        yield from _inner_iter_nested(tuple(data[k] for data in data_tuple))
    else:
      yield data_tuple

  return _inner_iter_nested(data_tuple)


def map_nested_tuple(
  data_tuple: Tuple[Any], op: Callable, *args, sortkey: bool = False, **kwargs
) -> Tuple[Any]:
  """A nested version of map function. Similar to map_nested
  but it iterates each of each nested data in the input tuple.
  Args:
    data_tuple (Tuple[Any]): A tuple of nested data.
    op (Callable): A function operate on each data
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """
  if not callable(op):
    raise ValueError("`op` must be a callable")
  if not isinstance(data_tuple, tuple):
    raise TypeError("`data_tuple` only accepts tuple, " f"got {type(data_tuple)}")

  def _inner_map_nested(data_tuple):
    if isinstance(data_tuple[0], dict):
      keys = data_tuple[0].keys()
      keys = sorted(keys) if sortkey else keys
      return {k: _inner_map_nested(tuple(data[k] for data in data_tuple)) for k in keys}
    elif isinstance(data_tuple[0], tuple):
      return tuple(
        _inner_map_nested(tuple(data[idx] for data in data_tuple))
        for idx in range(len(data_tuple[0]))
      )
    else:
      return op(data_tuple, *args, **kwargs)

  return _inner_map_nested(data_tuple)


def nested_to_numpy(
  data: Any, dtype: Optional[np.dtype] = None, sortkey: bool = False
) -> Any:
  """Convert all items in a nested data into
  numpy arrays
  Args:
    data (Any): A nested data
    dtype (np.dtype): data type. Defaults to None.
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  Returns:
    Any: A nested data same as `data`
  """
  op = lambda arr: np.asarray(arr, dtype=dtype)
  return map_nested(data, op, sortkey=sortkey)


def nested_to_jaxnumpy(
  data: Any, dtype: Optional[jnp.dtype] = None, sortkey: bool = False
) -> Any:
  """Convert all items in a nested data into
  Args:
    data (Any): a nested data
    dtype (jnp.dtype): data type. Defaults to None.
    sortkey (bool, optional): whether to sort dict's key.
      Defaults to False.
  """
  # Avoid circular import
  op = lambda arr: jnp.asarray(arr, dtype=dtype)
  return map_nested(data, op, sortkey=sortkey)


def unpack_structure(data: Any, sortkey: bool = False) -> Tuple[Any, List[Any]]:
  """Extract structure and flattened data from a nested data
  For example:
    >>> data = {'a': 'abc', 'b': (2.0, [3, 4, 5])}
    >>> struct, flat_data = extract_struct(data)
    >>> flat_data
    ['abc', 2.0, [3, 4, 5]]
    >>> struct
    {'a': 0, 'b': (1, 2)}

  Args:
    data (Any): A nested data
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """
  _count_op = lambda v, c: next(c)
  counter = itertools.count(0)
  struct = map_nested(data, _count_op, counter, sortkey=sortkey)
  size = next(counter)
  flat_data = [None] * size

  def _flat_op(ind_and_data, flat_data):
    ind, data = ind_and_data
    flat_data[ind] = data

  map_nested_tuple((struct, data), _flat_op, flat_data, sortkey=sortkey)
  return struct, flat_data


def pack_sequence(struct: Any, flat_data: List[Any], sortkey: bool = False) -> Any:
  """An inverse operation of `extract_structure`
  Args:
    struct (Any): A nested structure each data field contains
      an index of elements in `flat_data`
    flat_data (List[Any]): flattened data.
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """
  _struct_op = lambda ind, flat: flat[ind]
  data = map_nested(struct, _struct_op, flat_data, sortkey=sortkey)
  return data
