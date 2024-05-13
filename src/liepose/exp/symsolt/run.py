import os
from typing import List

from jaxlie import R3SO3, SE3
from tqdm import tqdm

from liepose.data.symsolt.dataset import register_dataset
from liepose.dist import NormalR3SO3, NormalSE3
from liepose.exp.symsolt.testbed import Testbed
from liepose.metrics import r3so3 as r3so3_metrics
from liepose.metrics import se3 as se3_metrics
from liepose.utils import datadir, ntfy_log
from liepose.utils.config import get_args, load_config_from_args, save_config
from liepose.utils.exploration import Explorations, Options

register_dataset("train-45k", datadir("symsolt/train-45k/"))
register_dataset("test-5k", datadir("symsolt/test-5k/"))


class TestbedSE3(Testbed):
  Lie = SE3
  LieDist = NormalSE3
  lie_metrics = se3_metrics


class TestbedR3SO3(Testbed):
  Lie = R3SO3
  LieDist = NormalR3SO3
  lie_metrics = r3so3_metrics


def get_testbed(lie_type: str):
  lie_type = lie_type.lower()
  assert lie_type in ["se3", "r3so3"], lie_type
  if lie_type == "se3":
    return TestbedSE3
  else:
    return TestbedR3SO3


def build_explorations() -> Explorations:
  explorations = Explorations()

  # datasets
  explorations.shape_names = [["tet", "cube", "icosa", "cone", "cyl"]]
  explorations.color_mode = ["none"]
  explorations.train_dataset = ["train-45k"]
  explorations.test_dataset = ["test-5k"]
  # model
  explorations.lie_type = ["se3"]
  explorations.image_res = [224]
  explorations.repr_type = ["tan"]
  explorations.resnet_depth = [34]
  explorations.mlp_layers = [1]
  explorations.fourier_block = [True]
  explorations.activ_fn = ["leaky_relu"]
  # training
  explorations.learn_noise = [True]
  explorations.loss_name = ["chordal"]
  explorations.train_steps = [800000]
  explorations.batch_size = [16]
  explorations.n_slices = [256]
  explorations.ckpt_steps = [100000]
  # learning rate
  explorations.init_lr = [1e-4]
  explorations.lr_decay_start = [0.5]
  explorations.lr_decay_steps = [0.5]
  explorations.lr_decay_rate = [0.1]
  explorations.end_lr = [1e-5]
  # ema
  explorations.ema_enabled = [True]
  explorations.ema_steps = [5]
  explorations.ema_tau = [0.999]
  # noise schedule
  explorations.noise_type = ["power"]
  explorations.noise_start = [1e-8]
  explorations.noise_end = [1.0]
  explorations.power = [3]
  explorations.timesteps = [100]
  # testing
  explorations.restore_step = [None]
  explorations.restore_path = [None]
  explorations.skip_train = [False]
  explorations.skip_test = [False]
  explorations.seed = [42]

  # sampling
  sample_options = Explorations()
  explorations.sample_options = sample_options

  sample_options.use_ddpm = [False]
  sample_options.eta = [1.0]
  sample_options.beta_skip = [True]
  sample_options.steps = [100]  # [100, 50, 10, 5, 1]
  sample_options.batch_size = [100]
  sample_options.n_slices = [1]

  return explorations


def run_experiment(a: Options, exp_path: str):
  sample_options_exps: Explorations = a.pop("sample_options")

  exp = get_testbed(a.lie_type)(a, exp_path, setup=True)

  if not a.skip_train:
    save_config(a, os.path.join(exp_path, "configs.yaml"))
    exp.run_train()
    ntfy_log(f"Training done: {exp_path}")
  else:
    print("[Skip training]")

  sample_options_exps: List[Options] = sample_options_exps.expand(
    excludes=[{"use_ddpm": True, "eta": 1.0}]
  )

  # testing
  if not a.skip_test:
    for sample_options in sample_options_exps:
      opt_name = sample_options.to_string(
        includes=["use_ddpm", "eta", "beta_skip", "steps"]
      )
      global_step = exp.global_step
      opt_path = os.path.join(exp_path, f"inference_{global_step}", opt_name)

      if sample_options.steps > a.timesteps:
        continue
      save_config(sample_options, os.path.join(opt_path, "configs.yaml"))
      exp.run_test(sample_options, opt_path)
    ntfy_log(f"Testing done: {exp_path}")
  else:
    print(" [Skip testing]")


def main():
  a = get_args(exp_name="symsolt-score-flat", desc="SymsolT experiment")
  root_path = a.root_path

  explorations = load_config_from_args(a, build_explorations())

  # expand experiment options
  explorations: List[Options] = explorations.expand()

  for exploration in tqdm(explorations, desc="Explorations"):
    exp_name = exploration.to_string(
      includes=["lie_type", "timesteps", "repr_type", "seed"]
    )
    exp_path = os.path.join(root_path, exp_name)
    run_experiment(exploration, exp_path)


if __name__ == "__main__":
  main()
