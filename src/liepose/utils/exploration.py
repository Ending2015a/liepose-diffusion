import itertools
from typing import List, Union

from .common import AttrDict


def exclude(opt, excludes):
  for exc in excludes:
    if exc.items() <= opt.items():
      return True
  return False


class Options(AttrDict):
  def to_string(
    self, excludes: List[str] = [], includes: List[str] = [], sep: str = "+"
  ) -> str:
    string = []
    for key, value in self.items():
      if key in excludes:
        continue
      if (len(includes) > 0) and (key not in includes):
        continue
      if callable(value):
        name = value.__name__
      elif not isinstance(value, str):
        name = str(value)
      else:
        name = value
      string.append(f"{key}={name}")
    return sep.join(string)

  def merge(self, opts: "Options", assert_keys: bool = True) -> "Options":
    assert isinstance(opts, Options), type(exps)

    for key, value in opts.items():
      if (assert_keys) and (key not in self.keys()):
        raise ValueError(f"Unknown options: {key}")

      if isinstance(self[key], Explorations):
        # in order to merge to self[key] (Explorations)
        # value should be a dict
        assert isinstance(value, dict), type(value)
        if not isinstance(value, (Options, Explorations)):
          value = Options(value)

        self[key].merge(value, assert_keys=assert_keys)
      elif isinstance(self[key], Options):
        # in order to merge to self[key] (Options)
        # value should be a dict
        assert isinstance(value, dict), type(value)
        assert not isinstance(value, Explorations)
        value = Options(value)
        self[key].merge(value, assert_keys=assert_keys)
      else:
        self[key] = value
    return self


class Explorations(AttrDict):
  def expand(self, excludes=[]) -> List["Options"]:
    key_list = list(self.keys())
    arg_list = list(self.values())

    # ensure all of the args are a list or a tuple
    for idx, arg in enumerate(arg_list):
      if isinstance(arg, Explorations):
        # protect nested
        arg_list[idx] = arg = [arg]
      assert isinstance(arg, (list, tuple)), key_list[idx]

    # expand lists
    explorations_options = itertools.product(*arg_list)

    explorations = []
    for options in explorations_options:
      opt = dict(zip(key_list, options))
      if exclude(opt, excludes):
        continue
      explorations.append(Options(opt))

    return explorations

  def merge(
    self, exps: Union[Options, "Explorations"], assert_keys: bool = True
  ) -> "Explorations":
    assert isinstance(exps, (Options, Explorations)), type(exps)

    _RootType = type(exps)

    for key, value in exps.items():
      if (assert_keys) and (key not in self.keys()):
        raise ValueError(f"Unknown options: {key}")

      if isinstance(self[key], Explorations):
        # in order to merge to self[key] (Explorations)
        # value should be a dict
        assert isinstance(value, dict), type(value)
        if not isinstance(value, (Options, Explorations)):
          # follow the root's type
          value = _RootType(value)

        self[key].merge(value, assert_keys=assert_keys)
      else:
        if isinstance(exps, Options):
          self[key] = [value]
        else:
          self[key] = value
    return self


if __name__ == "__main__":
  # base
  base_exps = Explorations(
    {
      "arg1": [1, 2, 3],
      "arg2": ["foo"],
      "exp": Explorations({"arg3": [1.0, 2.0]}),
      "arg4": [{"a": 1}, {"b": 1}],
    }
  )

  # file
  file_exps = Options({"exp": {"arg3": 1.0}, "arg4": {"a": 1}})

  # cmd
  cmd_exps = Explorations(
    {
      "arg1": [4, 5],
      "exp": {"arg3": [3.0, 4.0]},
    }
  )

  base_exps.merge(file_exps)
  base_exps.merge(cmd_exps)

  exps = base_exps.expand()

  for exp in exps:
    print(exp)
    print(exp.to_string(excludes=["arg1"], includes=["arg1", "arg4"]))
