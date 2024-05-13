import os

import jax
import jax.numpy as jnp
import numpy as np

from liepose.metrics import so3 as so3_metrics
from liepose.utils import nest

from ..base import BaseEvaluator
from ..utils import write_file

REPORT_ROT_THRES = [2, 5, 10]


@jax.jit
def metric_rot(r_gt, r_pred):
  return jax.vmap(
    lambda rp: jnp.min(
      jax.vmap(lambda gt: jnp.sqrt(so3_metrics.geodesic_distance(gt, rp)))(r_gt)
    )
  )(r_pred)


@jax.jit
def get_metrics(r_gt, r_pred, label, time):
  t_dim = r_pred.shape[0]
  s_dim = r_pred.shape[1]
  r_pred = r_pred.reshape((t_dim * s_dim, -1))

  rot = metric_rot(r_gt, r_pred).reshape((t_dim, s_dim))  # (t, s)
  label = label * jnp.ones_like(rot, dtype=jnp.int32)  # (t, s)
  time = time * jnp.ones_like(rot, dtype=jnp.float32)
  return {"rot": rot, "id": label, "time": time}


def get_metrics_summary(rot):
  N = rot.shape[0]
  m_rot = np.nanmean(rot)

  res = {"rot": m_rot, "rot(deg)": np.degrees(m_rot)}

  for th in REPORT_ROT_THRES:
    rot_th = np.sum(rot <= np.radians(th)) / N * 100.0
    res[f"rot_{th}"] = rot_th

  return res


def get_summary(metrics):
  rot = np.array(metrics["rot"])
  ids = np.array(metrics["id"])
  time = np.array(metrics["time"])

  m = get_metrics_summary(rot)

  # report per shape
  for id in np.unique(ids):
    m_per_shape = get_metrics_summary(rot[ids == id])

    for key in m_per_shape.keys():
      m[key + f"_id{id}"] = m_per_shape[key]

  m["time"] = np.nanmean(time)
  return m


class Evaluator(BaseEvaluator):
  def __init__(self, a, path: str, exp):
    self.a = a
    self.path = path
    self.exp = exp
    self._predictions = []

  def reset(self):
    self._predictions = []

  def process(self, batch_data, outputs):
    # outputs:
    #   seq_r0: sequence of r0 (t, b, s, 4)
    #   seq_rt: sequence of rt (t, b, s, 4)
    #   time: inference time
    b_dim = outputs["seq_r0"].shape[1]

    assert "rot_gt" in batch_data, batch_data.keys()
    assert "label" in batch_data, batch_data.keys()

    for b_idx in range(b_dim):
      rot_gt = batch_data["rot_gt"][b_idx]
      label = batch_data["label"][b_idx]

      seq_r0 = np.asarray(outputs["seq_r0"][:, b_idx])  # (t, s, 4)
      seq_rt = np.asarray(outputs["seq_rt"][:, b_idx])  # (t, s, 4)
      time = outputs["time"]

      prediction = {
        "seq_r0": seq_r0,
        "seq_rt": seq_rt,
        "rot_gt": rot_gt,
        "label": label,
        "time": time,
      }

      self._predictions.append(prediction)

  def summarize(self):
    r0_metrics = []
    rt_metrics = []
    for pred in self._predictions:
      seq_r0 = jnp.array(pred["seq_r0"], dtype=jnp.float32)  # (t, s, 4)
      seq_rt = jnp.array(pred["seq_rt"], dtype=jnp.float32)  # (t, s, 4)
      rot_gt = jnp.array(pred["rot_gt"], dtype=jnp.float32)  # (n, 3, 3)
      label = jnp.array(pred["label"], dtype=jnp.int32)
      time = jnp.array(pred["time"], dtype=jnp.float32)

      # calculate metrics per sample
      r0_met = get_metrics(rot_gt, seq_r0, label, time)
      rt_met = get_metrics(rot_gt, seq_rt, label, time)

      r0_metrics.append(nest.map_nested(r0_met, lambda x: np.asarray(x)))
      rt_metrics.append(nest.map_nested(rt_met, lambda x: np.asarray(x)))

    # concat all samples
    op = lambda xs: np.concatenate(xs, axis=1)
    r0_metrics = nest.map_nested_tuple(tuple(r0_metrics), op=op)  # (t, bs)
    rt_metrics = nest.map_nested_tuple(tuple(rt_metrics), op=op)  # (t, bs)

    # summarize by time
    r0_summary_by_time = []
    rt_summary_by_time = []
    t_dim = next(iter(nest.iter_nested(r0_metrics))).shape[0]
    for t in range(t_dim):
      op = lambda xs, t=t: xs[t]
      r0_summary = get_summary(nest.map_nested(r0_metrics, op=op))
      rt_summary = get_summary(nest.map_nested(rt_metrics, op=op))
      op = lambda x: np.array(x).item()
      r0_summary_by_time.append(nest.map_nested(r0_summary, op=op))
      rt_summary_by_time.append(nest.map_nested(rt_summary, op=op))

    summary = dict(
      final_metrics=rt_summary_by_time[-1],
      min_metrics=min(rt_summary_by_time, key=lambda m: m["rot"]),
      final_metrics_r0=r0_summary_by_time[-1],
      min_metrics_r0=min(r0_summary_by_time, key=lambda m: m["rot"]),
      initial_metrics=rt_summary_by_time[0],
    )

    write_file(os.path.join(self.path, "summary.json"), summary, indent=2)
    write_file(os.path.join(self.path, "r0_metrics.json"), r0_summary_by_time, indent=2)
    write_file(os.path.join(self.path, "rt_metrics.json"), rt_summary_by_time, indent=2)
