import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed):
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)


def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def repeat_dataloader(dataloader: DataLoader):
  while True:
    yield from dataloader


def write_file(path, data, **kwargs):
  if path.endswith(".json"):
    with open(path, "w") as f:
      json.dump(data, f, **kwargs)
  elif path.endswith(".pkl"):
    import pickle

    with open(path, "wb") as f:
      pickle.dump(data, f, **kwargs)
  elif path.endswith(".gz"):
    import gzip
    import pickle

    with gzip.open(path, "wb") as f:
      pickle.dump(data, f, **kwargs)
  else:
    raise ValueError(f"Unrecognized file extension [.json, .pkl, .gz]: {path}")


def load_file(path, **kwargs):
  if path.endswith(".json"):
    with open(path) as f:
      data = json.load(f, **kwargs)
  elif path.endswith(".pkl"):
    import pickle

    with open(path, "rb") as f:
      data = pickle.load(f, **kwargs)
  elif path.endswith(".gz"):
    import gzip
    import pickle

    with gzip.open(path, "rb") as f:
      data = pickle.load(f, **kwargs)
  else:
    raise ValueError(f"Unrecognized file extension [.json, .pkl, .gz]: {path}")
  return data
