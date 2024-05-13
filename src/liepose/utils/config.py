import argparse
import os
from typing import Dict

from omegaconf import OmegaConf

from .exploration import Explorations, Options


def get_args(exp_name="exp1", desc="Liepose"):
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument("--root_path", type=str, default=f"./logs/experiments/{exp_name}")
  parser.add_argument(
    "--config", type=str, default=None, help="restore arguments from path"
  )
  parser.add_argument(
    "--config_explore", type=str, default=None, help="load explorations from file"
  )
  parser.add_argument("dot_list", nargs=argparse.REMAINDER)

  return parser.parse_args()


def merge_config(explorations, conf: OmegaConf, Type=Explorations) -> Explorations:
  conf = OmegaConf.to_container(conf, resolve=True)
  explorations.merge(Type(conf))
  return explorations


def load_config_from_args(a, explorations: Explorations) -> Explorations:
  # load config from yaml file
  if a.config is not None:
    explorations = merge_config(explorations, OmegaConf.load(a.config), Options)
  # load explorations from yaml file
  if a.config_explore is not None:
    explorations = merge_config(explorations, OmegaConf.load(a.config_explore))
  # load config from dot list
  explorations = merge_config(explorations, OmegaConf.from_dotlist(a.dot_list))
  return explorations


def save_config(a: Dict, path: str):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  OmegaConf.save(config=OmegaConf.create(dict(a)), f=path)
