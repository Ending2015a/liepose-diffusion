# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague
# modified from eval_bop19.py
"""Evaluation script for the BOP Challenge 2019."""

"""
python3 lib/pysixd/scripts/eval_pose_results_more.py \
  --results_path data/BOP_DATASETS/lm_full/test/my_val_initial_poses_bb8_split/ \
  --result_filenames my-val_lm-test-bb8.csv \
  --targets_filename lm_test_targets_bb8.json \
  --error_types ad,proj,rete,vsd,mssd,mspd \
  --renderer_type python # egl, cpp, aae, python

python3 lib/pysixd/scripts/eval_pose_results_more.py \
  --results_path data/BOP_DATASETS/lm_full/test/PoseCNN_val_cup_bowl_my_val_initial_poses_bb8_split/  \
  --result_filenames  PoseCNN-val-cup-bowl-my-val_lm-test-bb8.csv    \
  --targets_filename lm_test_targets_bb8.json     \
  --error_types ad,proj,rete,vsd,mssd,mspd     \
  --renderer_type python # egl, cpp, aae, python
"""
import copy
import os
import subprocess
import time

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

os.environ["PYOPENGL_PLATFORM"] = "egl"
cur_dir = os.path.dirname(os.path.abspath(__file__))
from bop_toolkit_lib import config, inout, misc

from liepose.utils import datadir, nest


# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
def get_default_parameters():
  p = {
    # Errors to calculate.
    "errors": [
      {
        "n_top": -1,
        "type": "mspd",
        "correct_th": [[th] for th in np.arange(5, 51, 5)],
      },
      {
        "n_top": -1,
        "type": "mssd",
        "correct_th": [[th] for th in np.arange(0.05, 0.51, 0.05)],
      },
      {
        "n_top": -1,
        "type": "vsd",
        "vsd_deltas": {
          "hb": 15,
          "hbs": 15,
          "icbin": 15,
          "icmi": 15,
          "itodd": 5,
          "lm": 15,
          "lmo": 15,
          "ruapc": 15,
          "tless": 15,
          "tudl": 15,
          "tyol": 15,
          "ycbv": 15,
          "hope": 15,
        },
        "vsd_taus": list(np.arange(0.05, 0.51, 0.05)),
        "vsd_normalized_by_diameter": True,
        "correct_th": [[th] for th in np.arange(0.05, 0.51, 0.05)],
      },
      {
        "n_top": -1,
        "type": "add",
        "correct_th": [[th] for th in [0.02, 0.05, 0.1]],
      },
      {
        "n_top": -1,
        "type": "adi",
        "correct_th": [[th] for th in [0.02, 0.05, 0.1]],
      },  # diameter
      {
        "n_top": -1,
        "type": "ad",  # adi for symmetric objects, add for normal objects
        "correct_th": [[th] for th in [0.02, 0.05, 0.1]],  # diameter
      },
      ##################
      # ADD(-S) with absolute threshold 2cm, for YCB-Video
      # {"n_top": -1, "type": "ABSadd", "correct_th": [[th] for th in [2]]},
      # {"n_top": -1, "type": "ABSadi", "correct_th": [[th] for th in [2]]},
      # # ABSadi for symmetric objects, ABSadd for normal objects
      # {"n_top": -1, "type": "ABSad", "correct_th": [[th] for th in [2]]},
      #################
      # AUC of ADD(-S) with a maximum distance 10cm, for YCB-Video, it uses the VOC 11 points method
      # {
      #   "n_top": -1,
      #   "type": "AUCadd",
      #   "correct_th": [[th] for th in np.linspace(10 / 10, 10, num=10)],
      # },
      # {
      #   "n_top": -1,
      #   "type": "AUCadi",
      #   "correct_th": [[th] for th in np.linspace(10 / 10, 10, num=10)],
      # },
      # # AUCadi for symmetric objects, AUCadd for normal objects
      # {
      #   "n_top": -1,
      #   "type": "AUCad",
      #   "correct_th": [[th] for th in np.linspace(10 / 10, 10, num=10)],
      # },
      ##################
      {
        "n_top": -1,
        "type": "re",
        "correct_th": [[th] for th in [2, 5, 10]],
      },  # deg
      {
        "n_top": -1,
        "type": "reS",
        "correct_th": [[th] for th in [2, 5, 10]],
      },  # deg
      {
        "n_top": -1,
        "type": "te",
        "correct_th": [[th] for th in [2, 5, 10]],
      },  # cm
      {
        "n_top": -1,
        "type": "tex",
        "correct_th": [[th] for th in [2, 5, 10]],
      },  # cm
      {
        "n_top": -1,
        "type": "tey",
        "correct_th": [[th] for th in [2, 5, 10]],
      },  # cm
      {
        "n_top": -1,
        "type": "tez",
        "correct_th": [[th] for th in [2, 5, 10]],
      },  # cm
      {
        "n_top": -1,
        "type": "teS",
        "correct_th": [[th] for th in [2, 5, 10]],
      },  # cm
      {
        "n_top": -1,
        "type": "teSx",
        "correct_th": [[th] for th in [2, 5, 10]],
      },  # cm
      {
        "n_top": -1,
        "type": "teSy",
        "correct_th": [[th] for th in [2, 5, 10]],
      },  # cm
      {
        "n_top": -1,
        "type": "teSz",
        "correct_th": [[th] for th in [2, 5, 10]],
      },  # cm
      {
        "n_top": -1,
        "type": "rete",
        "correct_th": [[2, 2], [5, 5], [10, 10]],
      },  # deg, cm
      {
        "n_top": -1,
        "type": "reteS",
        "correct_th": [[2, 2], [5, 5], [10, 10]],
      },  # deg, cm
      {
        "n_top": -1,
        "type": "proj",
        "correct_th": [[th] for th in [2, 5, 10]],
      },  # pixel
      {
        "n_top": -1,
        "type": "projS",
        "correct_th": [[th] for th in [2, 5, 10]],
      },  # pixel
    ],
    # Minimum visible surface fraction of a valid GT pose.
    # -1 == k most visible GT poses will be considered, where k is given by
    # the "inst_count" item loaded from "targets_filename".
    "visib_gt_min": -1,
    # See misc.get_symmetry_transformations().
    "max_sym_disc_step": 0.01,
    # Type of the renderer (used for the VSD pose error function).
    "renderer_type": "cpp",  # Options: 'cpp', 'python', 'aae', 'egl'.
    # Names of files with results for which to calculate the errors (assumed to be
    # stored in folder p['results_path']). See docs/bop_challenge_2019.md for a
    # description of the format. Example results can be found at:
    # http://ptak.felk.cvut.cz/6DB/public/bop_sample_results/bop_challenge_2019/
    "result_filenames": ["/relative/path/to/csv/with/results"],
    # Folder with results to be evaluated.
    "results_path": config.results_path,
    # Folder for the calculated pose errors and performance scores.
    "eval_path": config.eval_path,
    # File with a list of estimation targets to consider. The file is assumed to
    # be stored in the dataset folder.
    "targets_filename": "test_targets_bop19.json",
  }
  return copy.deepcopy(p)


################################################################################


def get_object_nums_from_targets(targets_path):
  targets = inout.load_json(targets_path)

  obj_nums_dict = {}
  for target in targets:
    obj_id = target["obj_id"]
    if obj_id not in obj_nums_dict:
      obj_nums_dict[obj_id] = 0
    obj_nums_dict[obj_id] += target["inst_count"]
  res_obj_nums_dict = {
    str(key): obj_nums_dict[key] for key in sorted(obj_nums_dict.keys())
  }
  return res_obj_nums_dict


def create_tabulate(evaluation_result):
  error_dicts = evaluation_result[0]
  headers = ["objects"]
  obj_recalls = {}
  avg_recalls = []
  for error_type, error_dict in error_dicts.items():
    headers.append(error_type)
    obj_count = error_dict["obj_count"]
    recalls = []
    counts = []
    for obj_id, recall in error_dict["obj_recalls"].items():
      if obj_id not in obj_recalls.keys():
        obj_recalls[obj_id] = []
      obj_recalls[obj_id].append(f"{recall*100:.2f}")
      recalls.append(recall)
      counts.append(obj_count[obj_id])
    if error_type.split("_")[0] in ["mspd", "mssd", "vsd"]:
      counts = np.array(counts)
      total_counts = np.sum(counts)
      weights = counts / total_counts
      avg_recall = np.sum(np.array(recalls) * weights)
      avg_recalls.append(f"{avg_recall*100:.2f}")
    else:
      avg_recalls.append(f"{np.mean(recalls)*100:.2f}")
  num_classes = len(obj_recalls)

  rows = []
  if num_classes > 0:
    rows.extend([[str(obj_id), *recalls] for obj_id, recalls in obj_recalls.items()])
  # avg
  rows.append([f"Avg({num_classes})", *avg_recalls])

  table = tabulate(
    rows,
    headers=headers,
    tablefmt="pipe",
    numalign="right",
    stralign="center",
    floatfmt=".2f",
  )
  return table


# Command line arguments.
# ------------------------------------------------------------------------------
def bop_evaluate(
  renderer_type: str = "cpp",
  result_filenames: str = None,
  results_path: str = None,
  dataset: str = "tless",
  eval_path: str = None,
  targets_filename: str = None,
  error_types: str = None,
  n_top: int = -1,
  score_only: bool = False,
  quiet: int = 0,
):
  misc.set_quiet(quiet)
  bop_datasets_path = datadir("bop_datasets")
  p = get_default_parameters()

  if not renderer_type:
    renderer_type = p["renderer_type"]
  if not result_filenames:
    result_filenames = ",".join(p["result_filenames"])
  if not results_path:
    results_path = p["results_path"]
  if not eval_path:
    eval_path = p["eval_path"]
  if not targets_filename:
    targets_filename = p["targets_filename"]
  if not error_types:
    error_types = ",".join(["mspd", "mssd", "vsd", "ad", "teS", "reS"])

  p["renderer_type"] = str(renderer_type)
  p["result_filenames"] = result_filenames.split(",")
  p["results_path"] = str(results_path)
  p["eval_path"] = str(eval_path)
  p["targets_filename"] = str(targets_filename)
  p["error_types"] = error_types.split(",")
  p["n_top"] = n_top

  evaluation_results = []

  # Evaluation.
  # ------------------------------------------------------------------------------
  for result_filename in p["result_filenames"]:
    results_dict = {}

    misc.log("===========")
    misc.log(f"EVALUATING: {result_filename}")
    misc.log("===========")

    time_start = time.time()

    # Volume under recall surface (VSD) / area under recall curve (MSSD, MSPD; AUCadd, AUCadi, AUCad).
    average_recalls = {}

    # Name of the result and the dataset.
    result_name = os.path.splitext(os.path.basename(result_filename))[0]
    if dataset is None:
      dataset = str(result_name.split("_")[1].split("-")[0])

    # logger.set_logger_dir(os.path.join(p["eval_path"], result_name), action="k")

    # Calculate the average estimation time per image.
    ests = inout.load_bop_results(
      os.path.join(p["results_path"], result_filename), version="bop19"
    )
    times = {}
    times_available = True
    for est in ests:
      result_key = "{:06d}_{:06d}".format(est["scene_id"], est["im_id"])
      if est["time"] < 0:
        # All estimation times must be provided.
        times_available = False
        break
      elif result_key in times:
        if abs(times[result_key] - est["time"]) > 0.001:
          raise ValueError(
            "The running time for scene {} and image {} is not the same for "
            "all estimates.".format(est["scene_id"], est["im_id"])
          )
      else:
        times[result_key] = est["time"]

    if times_available:
      average_time_per_image = np.mean(list(times.values()))
    else:
      average_time_per_image = -1.0

    base_path = os.path.join(bop_datasets_path, dataset)
    obj_count = get_object_nums_from_targets(
      os.path.join(base_path, p["targets_filename"])
    )

    # Evaluate the pose estimates.
    for error in tqdm(p["errors"], desc="Evaluation"):
      if error["type"] not in p["error_types"]:
        continue
      # Calculate error of the pose estimates.
      # NOTE: allow change default n_top if do not want n_top = -1
      # SISO setting: n_top=1
      if error["n_top"] != p["n_top"]:
        error["n_top"] = p["n_top"]
      misc.log("n_top: {}".format(error["n_top"]))
      calc_errors_cmd = [
        "python3",
        os.path.join(cur_dir, "eval_calc_errors.py"),
        "--n_top={}".format(error["n_top"]),
        "--error_type={}".format(error["type"]),
        f"--result_filenames={result_filename}",
        f"--datasets_path={bop_datasets_path}",
        "--renderer_type={}".format(p["renderer_type"]),
        "--results_path={}".format(p["results_path"]),
        "--eval_path={}".format(p["eval_path"]),
        "--targets_filename={}".format(p["targets_filename"]),
        "--max_sym_disc_step={}".format(p["max_sym_disc_step"]),
        "--skip_missing=1",
        f"--quiet={quiet}",
      ]
      if error["type"] == "vsd":
        vsd_deltas_str = ",".join([f"{k}:{v}" for k, v in error["vsd_deltas"].items()])
        calc_errors_cmd += [
          f"--vsd_deltas={vsd_deltas_str}",
          "--vsd_taus={}".format(",".join(map(str, error["vsd_taus"]))),
          "--vsd_normalized_by_diameter={}".format(error["vsd_normalized_by_diameter"]),
        ]
      if not score_only:
        misc.log("Running: " + " ".join(calc_errors_cmd))
        if subprocess.call(calc_errors_cmd) != 0:
          raise RuntimeError("Calculation of pose errors failed.")

      # Paths (rel. to p['eval_path']) to folders with calculated pose errors.
      # For VSD, there is one path for each setting of tau. For the other pose
      # error functions, there is only one path.
      error_dir_paths = {}
      if error["type"] == "vsd":
        for vsd_tau in error["vsd_taus"]:
          error_sign = misc.get_error_signature(
            error["type"],
            error["n_top"],
            vsd_delta=error["vsd_deltas"][dataset],
            vsd_tau=vsd_tau,
          )
          error_dir_paths[error_sign] = os.path.join(result_name, error_sign)
      else:
        error_sign = misc.get_error_signature(error["type"], error["n_top"])
        error_dir_paths[error_sign] = os.path.join(result_name, error_sign)

      # Recall scores for all settings of the threshold of correctness (and also
      # of the misalignment tolerance tau in the case of VSD).
      recalls = []

      # Calculate performance scores.
      error_results_dicts = []
      for error_sign, error_dir_path in error_dir_paths.items():
        for correct_th in error["correct_th"]:
          calc_scores_cmd = [
            "python3",
            os.path.join(cur_dir, "eval_calc_scores.py"),
            f"--error_dir_paths={error_dir_path}",
            f"--datasets_path={bop_datasets_path}",
            "--eval_path={}".format(p["eval_path"]),
            "--targets_filename={}".format(p["targets_filename"]),
            "--visib_gt_min={}".format(p["visib_gt_min"]),
            f"--quiet={quiet}",
          ]

          calc_scores_cmd += [
            "--correct_th_{}={}".format(error["type"], ",".join(map(str, correct_th)))
          ]

          misc.log("Running: " + " ".join(calc_scores_cmd))
          if subprocess.call(calc_scores_cmd) != 0:
            raise RuntimeError("Calculation of scores failed.")

          # Path to file with calculated scores.
          score_sign = misc.get_score_signature(correct_th, p["visib_gt_min"])

          scores_filename = f"scores_{score_sign}.json"
          scores_path = os.path.join(
            p["eval_path"], result_name, error_sign, scores_filename
          )

          # Load the scores.
          misc.log(f"Loading calculated scores from: {scores_path}")
          scores = inout.load_json(scores_path)
          recalls.append(scores["recall"])

          scores["obj_count"] = copy.deepcopy(obj_count)

          if error["type"] in ["mspd", "mssd", "vsd"]:
            error_results_dicts.append(copy.deepcopy(scores))
          else:
            result_key = f"{error['type']}_{correct_th[0]}"
            results_dict[result_key] = copy.deepcopy(scores)

      if error["type"] in ["mspd", "mssd", "vsd"]:
        error_results = nest.map_nested_tuple(
          tuple(error_results_dicts), op=lambda xs: np.mean(xs).item()
        )
        s_th = error["correct_th"][0][0]
        e_th = error["correct_th"][-1][0]
        result_key = f"{error['type']}_{s_th}:{e_th}"
        results_dict[result_key] = error_results

      average_recalls[error["type"]] = np.mean(recalls)

      misc.log(
        "error_type: {} thresholds: {}".format(
          error["type"], " ".join(map(str, error["correct_th"]))
        )
      )
      misc.log("Recall scores: {}".format(" ".join(map(str, recalls))))
      misc.log("Average recall: {}".format(average_recalls[error["type"]]))
    #########################################################
    time_total = time.time() - time_start
    misc.log(f"Evaluation of {result_filename} took {time_total}s.")

    # Calculate the final scores.
    final_scores = {}
    for error in p["errors"]:
      if error["type"] not in p["error_types"]:
        continue
      final_scores["bop19_average_recall_{}".format(error["type"])] = average_recalls[
        error["type"]
      ]

    # Final score for the given dataset.
    if all(_e_type in p["error_types"] for _e_type in ["mspd", "mssd", "vsd"]):
      final_scores["bop19_average_recall"] = np.mean(
        [
          average_recalls["mspd"],
          average_recalls["mssd"],
          average_recalls["vsd"],
        ]
      )

    # Average estimation time per image.
    final_scores["bop19_average_time_per_image"] = average_time_per_image

    # Save the final scores.
    final_scores_path = os.path.join(p["eval_path"], result_name, "scores_bop19.json")
    inout.save_json(final_scores_path, final_scores)

    # Print the final scores.
    misc.log("FINAL SCORES:")
    for score_name, score_value in sorted(final_scores.items()):
      misc.log(f"- {score_name}: {score_value}")
    misc.log(f"final score path {final_scores_path}")

    evaluation_results.append((results_dict, final_scores))

  misc.log("Done.")

  if len(evaluation_results) == 1:
    evaluation_results = evaluation_results[0]

  return evaluation_results
