import os

import jax
import jax.numpy as jnp
import numpy as np
from jaxlie import R3SO3, SO3

from liepose.metrics import r3so3 as r3so3_metrics
from liepose.metrics import se3 as se3_metrics
from liepose.metrics import so3 as so3_metrics
from liepose.utils import nest

from ..base import BaseEvaluator
from ..utils import write_file

REPORT_ADD_THRES = [0.02, 0.05, 0.1]
REPORT_ROT_THRES = [2, 5, 10]
REPORT_TRAN_THRES = [0.02, 0.05, 0.1]


@jax.jit
def metric_add(p_gt, p_pred):
  return jax.vmap(
    lambda pp: jnp.min(
      jax.vmap(lambda gt: jnp.mean(jnp.linalg.norm(pp - gt, axis=-1), axis=-1))(p_gt)
    )
  )(p_pred)


@jax.jit
def metric_rot(r_gt, r_pred):
  return jax.vmap(
    lambda rp: jnp.min(
      jax.vmap(
        lambda gt: jnp.sqrt(
          so3_metrics.geodesic_distance(
            r3so3_metrics.as_lie(gt).rotation(), r3so3_metrics.as_lie(rp).rotation()
          )
        )
      )(r_gt)
    )
  )(r_pred)


@jax.jit
def metric_tran(r_gt, r_pred):
  return jax.vmap(
    lambda rp: jnp.min(
      jax.vmap(
        lambda gt: jnp.linalg.norm(
          r3so3_metrics.as_lie(gt).translation()
          - r3so3_metrics.as_lie(rp).translation(),
          axis=-1,
        )
      )(r_gt)
    )
  )(r_pred)


@jax.jit
def metric_geo(r_gt, r_pred):
  return jax.vmap(
    lambda rp: jnp.min(
      jax.vmap(
        lambda gt: jnp.sqrt(
          se3_metrics.geodesic_distance(se3_metrics.as_lie(gt), se3_metrics.as_lie(rp))
        )
      )(r_gt)
    )
  )(r_pred)


@jax.jit
def transform_points(r, ps):
  return jax.vmap(lambda p: r3so3_metrics.as_lie(r).apply(p))(ps)


@jax.jit
def get_metrics(r_gt, r_pred, points, label, time):
  assert r_gt.shape[-1] == 7
  assert r_pred.shape[-1] == 7

  t_dim = r_pred.shape[0]
  s_dim = r_pred.shape[1]
  r_pred = r_pred.reshape((t_dim * s_dim, -1))

  p_gt = jax.vmap(lambda r, p=points: transform_points(r, p))(r_gt)
  p_pred = jax.vmap(lambda r, p=points: transform_points(r, p))(r_pred)

  add = metric_add(p_gt, p_pred).reshape((t_dim, s_dim))
  rot = metric_rot(r_gt, r_pred).reshape((t_dim, s_dim))
  tran = metric_tran(r_gt, r_pred).reshape((t_dim, s_dim))
  geo = metric_geo(r_gt, r_pred).reshape((t_dim, s_dim))

  label = label * jnp.ones_like(rot, dtype=jnp.int32)
  time = time * jnp.ones_like(rot, dtype=jnp.float32)
  return {"add": add, "rot": rot, "tran": tran, "geo": geo, "id": label, "time": time}


def get_metrics_summary(add, rot, tran, geo):
  N = rot.shape[0]
  m_add = np.nanmean(add)
  m_rot = np.nanmean(rot)
  m_tran = np.nanmean(tran)
  m_geo = np.nanmean(geo)
  std_add = np.nanstd(add)
  std_rot = np.nanstd(rot)
  std_tran = np.nanstd(tran)
  std_geo = np.nanstd(geo)

  res = {
    "add": m_add,
    "rot": m_rot,
    "rot(deg)": np.degrees(m_rot),
    "tran": m_tran,
    "geo": m_geo,
    "add_std": std_add,
    "rot_std": std_rot,
    "rot_std(deg)": np.degrees(std_rot),
    "tran_std": std_tran,
    "geo_std": std_geo,
  }

  for th in REPORT_ADD_THRES:
    add_th = np.sum(add <= th) / N * 100.0
    res[f"add_{th}"] = add_th

  for th in REPORT_ROT_THRES:
    rot_th = np.sum(rot <= np.radians(th)) / N * 100.0
    res[f"rot_{th}"] = rot_th

  for th in REPORT_TRAN_THRES:
    tran_th = np.sum(tran <= th) / N * 100.0
    res[f"tran_{th}"] = tran_th

  return res


def get_summary(metrics):
  add = np.array(metrics["add"])
  rot = np.array(metrics["rot"])
  tran = np.array(metrics["tran"])
  geo = np.array(metrics["geo"])
  ids = np.array(metrics["id"])
  time = np.array(metrics["time"])

  m = get_metrics_summary(add, rot, tran, geo)

  # report per shape
  for id in np.unique(ids):
    m_per_shape = get_metrics_summary(
      add[ids == id], rot[ids == id], tran[ids == id], geo[ids == id]
    )

    for key in m_per_shape.keys():
      m[key + f"_id{id}"] = m_per_shape[key]

  m["time"] = np.nanmean(time)
  return m


@jax.jit
def get_gt(rs, t):
  return jax.vmap(
    lambda r, t=t: R3SO3.from_rotation_and_translation(SO3(r), t).wxyz_xyz
  )(rs)  # (n, 7)


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

    assert "rotations_equivalent" in batch_data, batch_data.keys()
    assert "translation" in batch_data, batch_data.keys()
    assert "label_shape" in batch_data, batch_data.keys()

    for b_idx in range(b_dim):
      rot_gt = jnp.array(batch_data["rotations_equivalent"][b_idx], dtype=jnp.float32)
      tran_gt = jnp.array(batch_data["translation"][b_idx], dtype=jnp.float32)
      points = jnp.array(batch_data["points"][b_idx], dtype=jnp.float32)
      label = jnp.array(batch_data["label_shape"][b_idx], dtype=jnp.int32)
      index = batch_data["index"][b_idx]

      seq_r0 = jnp.array(outputs["seq_r0"][:, b_idx], dtype=jnp.float32)
      seq_rt = jnp.array(outputs["seq_rt"][:, b_idx], dtype=jnp.float32)
      time = jnp.array(outputs["time"], dtype=jnp.float32)

      # calculate metrics per sample here to save memory
      r_gt = get_gt(rot_gt, tran_gt)
      r0_met = get_metrics(r_gt, seq_r0, points, label, time)
      rt_met = get_metrics(r_gt, seq_rt, points, label, time)
      r0_met = nest.map_nested(r0_met, lambda x: np.asarray(x))
      rt_met = nest.map_nested(rt_met, lambda x: np.asarray(x))

      prediction = {
        "r0_met": r0_met,
        "rt_met": rt_met,
        "index": index,
        "label": label,
        "time": time,
      }

      self._predictions.append(prediction)

  def summarize(self):
    r0_metrics = []
    rt_metrics = []
    for pred in self._predictions:
      r0_metrics.append(pred["r0_met"])
      rt_metrics.append(pred["rt_met"])

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
      min_metrics=min(rt_summary_by_time, key=lambda m: m["add"]),
      final_metrics_r0=r0_summary_by_time[-1],
      min_metrics_r0=min(r0_summary_by_time, key=lambda m: m["add"]),
      initial_metrics=rt_summary_by_time[0],
    )

    write_file(os.path.join(self.path, "summary.json"), summary, indent=2)
    write_file(os.path.join(self.path, "r0_metrics.json"), r0_summary_by_time, indent=2)
    write_file(os.path.join(self.path, "rt_metrics.json"), rt_summary_by_time, indent=2)
