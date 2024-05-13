import csv
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

from liepose.utils import nest
from liepose.utils.npfn import get_allo_rotation, get_obj_ray, wxyz_to_xyzw

from ..base import BaseEvaluator
from .bop_scripts import bop_evaluate, create_tabulate
from .dataset import BOPPoseDataset
from .registry import get_dataset


class Evaluator(BaseEvaluator):
  def __init__(
    self, a, path: str, exp, dataset: BOPPoseDataset, full_eval: bool = True
  ):
    self.a = a
    self.path = path
    self.exp = exp
    self.dataset_name = dataset.dataset.name
    self.dataset_info = get_dataset(self.dataset_name)
    self.dataset_type = self.dataset_info["dataset_type"]
    self.full_eval = full_eval

    self._predictions = []
    self._csv_path = os.path.join(self.path, f"inferences_{self.dataset_type}-test.csv")
    self._eval_path = os.path.join(self.path, "evaluations/")
    self._json_path = os.path.join(self.path, "evaluation_results.json")
    self._summ_path = os.path.join(self.path, "evaluation_summary.txt")

  def reset(self):
    self._predictions = []

  def _process_pose(self, inp, pred):
    # if output is in allocentric space,
    # here we post-process it to egocentric space

    pose = pred  # wxyz_xyz

    assert pose.shape == (7,), pose.shape

    quat = pose[:4]
    tran = pose[4:]
    rot = Rotation.from_quat(wxyz_to_xyzw(quat))
    mat = rot.as_matrix()

    # scaled translation -> allocentric translation
    if self.a.use_scaled_tran:
      tran_ratio = inp["translation_ratio"]
      tran_offset = inp["translation_offset"]
      proj_bbox_center = inp["proj_bbox_center"]

      tran = (tran + tran_offset) * tran_ratio
      tran[:2] = (tran[:2] + proj_bbox_center) * tran[-1]

      if self.a.use_reproj:
        tran = tran @ np.linalg.inv(inp["cam_crop"]).T
      else:
        tran = tran @ np.linalg.inv(inp["cam"]).T

    # calculate allocentric to egocentric transform
    if self.a.use_reproj:
      obj_ray = get_obj_ray(inp["cam"], inp["bbox_center"])
      ego2allo = get_allo_rotation(obj_ray)
    else:
      ego2allo = get_allo_rotation(tran)

    ego2allo = Rotation.from_quat(wxyz_to_xyzw(ego2allo))
    allo2ego_mat = ego2allo.inv().as_matrix()

    # allocentric translation -> egocentric translation
    if self.a.use_scaled_tran or self.a.use_allo_tran:
      if self.a.use_reproj:
        tran = tran @ allo2ego_mat.T

    # allocentric rotation -> egocentric rotation
    if self.a.use_allo_rot:
      mat = allo2ego_mat @ mat

    tran = tran * inp["scale_factor"]  # default: M2MM

    return mat, tran

  def process(self, inputs, outputs):
    # outputs:
    #   seq_r0: sequence of r0 (t, b, s, 4)
    #   seq_rt: sequence of rt (t, b, s, 4)
    b_dim = outputs["seq_r0"].shape[1]

    for b_idx in range(b_dim):
      inp = nest.map_nested(inputs, op=lambda x, idx=b_idx: x[idx])
      pred = outputs["seq_rt"][-1, b_idx, 0]
      rot, tran = self._process_pose(inp, pred)
      infer_time = inp.get("time", 0.0) + outputs.get("time", 0.0)

      prediction = {
        "scene_id": inp["scene_id"],
        "im_id": inp["image_id"],
        "obj_id": inp["object_id"],
        "score": inp.get("score", 1.0),
        "time": infer_time,
        "R": " ".join(str(r) for r in rot.flatten().tolist()),
        "t": " ".join(str(t) for t in tran.flatten().tolist()),
      }

      self._predictions.append(prediction)

  def _process_time(self):
    times = {}
    num_objects = {}
    for prediction in self._predictions:
      result_key = f"{prediction['scene_id']:06d}_{prediction['im_id']:06d}"
      if result_key in times:
        times[result_key] += prediction["time"]
        num_objects[result_key] += 1
      else:
        times[result_key] = prediction["time"]
        num_objects[result_key] = 1
    # calculate average time
    for key in times.keys():
      times[key] /= num_objects[key]

    for prediction in self._predictions:
      result_key = f"{prediction['scene_id']:06d}_{prediction['im_id']:06d}"
      prediction["time"] = times[result_key]

  def _save_bop_results(self, predictions):
    # save predictions in bop format
    line_head = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]

    bop_results = []
    for prediction in predictions:
      bop_results.append([prediction[key] for key in line_head])

    os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
    with open(self._csv_path, "w") as f:
      csv_writer = csv.writer(f, delimiter=",")
      csv_writer.writerow(line_head)

      for line in bop_results:
        csv_writer.writerow(line)

  def summarize(self):
    self._process_time()
    # save csv file
    self._save_bop_results(self._predictions)

  def evaluate(self):
    test_bop_path = self.dataset_info.get("test_bop_path", "test_targets_bop19.json")

    os.makedirs(self._eval_path, exist_ok=True)

    if self.full_eval:
      error_types = "mspd,mssd,vsd,ad,reS,teS"
    else:
      error_types = "ad,reS,teS"  # "teSx,teSy,teSz"

    # run evaluation scripts
    evaluation_result = bop_evaluate(
      renderer_type="cpp",
      result_filenames=os.path.abspath(self._csv_path),
      results_path="",
      dataset=self.dataset_type,
      eval_path=os.path.abspath(self._eval_path),
      targets_filename=test_bop_path,
      error_types=error_types,
      n_top=-1,
      score_only=False,
      quiet=1,
    )

    with open(self._json_path, "w") as f:
      json.dump(evaluation_result, f, indent=2, ensure_ascii=False)

    table = create_tabulate(evaluation_result)
    with open(self._summ_path, "w") as f:
      f.write(table)

    print(table)
