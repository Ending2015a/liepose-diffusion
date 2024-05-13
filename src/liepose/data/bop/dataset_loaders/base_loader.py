import copy
import itertools
from typing import Any, Dict, List

from tabulate import tabulate

from ..data_utils import get_default_cache_path, load_json
from ..registry import get_dataset, get_dataset_type


def load_dataset_by_name(dataset_name: str, **kwargs) -> "BOPDatasetLoader":
  dataset_info = copy.deepcopy(get_dataset(dataset_name))
  dataset_type = dataset_info.pop("dataset_type")
  dataset_type = get_dataset_type(dataset_type)
  # overwrite dataset info
  dataset_info.update(kwargs)

  # create dataset
  dataset = dataset_type(name=dataset_name, **dataset_info)
  return dataset


def load_datasets(datasets, **kwargs) -> List["BOPDatasetLoader"]:
  if not isinstance(datasets, (list, tuple)):
    datasets = [datasets]

  loaded_datasets = []
  for dataset in datasets:
    if isinstance(dataset, str):
      dataset = load_dataset_by_name(dataset, **kwargs)
    loaded_datasets.append(dataset)

  return loaded_datasets


def concat_datasets(datasets: List["BOPDatasetLoader"]) -> "BOPDatasetLoader":
  if not isinstance(datasets, (list, tuple)):
    datasets = [datasets]

  assert len(datasets) > 0

  if len(datasets) == 1:
    return datasets[0]

  dataset_dicts = []
  include_ids = []

  for dataset in datasets:
    if len(datasets) > 1:
      # TODO: find a more reliable way to check if two datasets
      # have the same models_info
      assert type(dataset) is type(
        datasets[0]
      ), f"{type(dataset).__name__} vs {type(datasets[0]).__name__}"
    if len(dataset.dataset_dicts) > 0:
      dataset_dicts.extend(dataset.dataset_dicts)
    if dataset.include_ids is not None:
      include_ids.extend(dataset.include_ids)
    # clear dataset dicts
    dataset.dataset_dicts = []

  if len(include_ids) == 0:
    include_ids = None
  else:
    include_ids = sorted(set(include_ids))

  new_dataset = BOPDatasetLoader(
    name=f"concat({','.join([dataset.name for dataset in datasets])})",
    path="./",
    models_info_path="./",
    include_ids=include_ids,
    width=dataset.width,
    height=dataset.height,
    short_names=dataset.short_names,
  )
  new_dataset.datasets = datasets
  new_dataset.dataset_dicts = dataset_dicts
  new_dataset.num_classes = dataset.num_classes
  new_dataset.dataset_type = dataset.dataset_type
  new_dataset.models_info = copy.deepcopy(dataset.models_info)
  new_dataset.objid2catid = copy.deepcopy(dataset.objid2catid)
  new_dataset.catid2objid = copy.deepcopy(dataset.catid2objid)

  return new_dataset


class BOPDatasetLoader:
  """
  Dataset type, this should match to the shorter name of BOP dataset.
  If this attr is not set, it will be set automatically when you
  register the dataset type with `register_dataset_type(type_name)`
  """

  dataset_type: str = None

  def __init__(
    self,
    name: str,
    path: str,
    models_info_path: str,
    detection_path: str = None,
    test_bop_path: str = None,
    include_ids: List[int] = None,
    cache_path: str = None,
    width: int = 640,
    height: int = 480,
    short_names: List[str] = None,
    num_scenes: int = None,
    filter_invalid: bool = True,
    det_thres: float = 0.0,
    det_topk_per_obj: int = 100,
    **kwargs,
  ):
    self.name = name
    self.path = path
    self.models_info_path = models_info_path
    self.detection_path = detection_path
    self.test_bop_path = test_bop_path
    self.include_ids = include_ids
    self.cache_path = cache_path
    self.width = width
    self.height = height
    self.aspect = width / height
    self.short_names = short_names
    self.num_scenes = num_scenes
    self.filter_invalid = filter_invalid
    self.det_thres = det_thres
    self.det_topk_per_obj = det_topk_per_obj

    if include_ids is not None:
      include_ids = sorted(set(include_ids))
    self.include_ids = include_ids

    if cache_path is None:
      hash_info = [
        self.name,
        self.path,
        self.models_info_path,
        self.test_bop_path,
        self.width,
        self.height,
      ]
      cache_path = get_default_cache_path(name, "dataset_dicts", hash_info)
    print(f"Dataset {name} cache path: {cache_path}")

    self.cache_path = cache_path

    """Object ID to models info"""
    self.models_info: Dict[int, Any] = {}
    """Object ID to category ID"""
    self.objid2catid: Dict[int, int] = {}
    """Category ID to object ID"""
    self.catid2objid: Dict[int, int] = {}

    self.dataset_dicts = []
    self.num_classes = 0

  def _load_detections(self):
    if self.detection_path is not None:
      self._load_detections_from_path()
    # flatten dataset dicts from per-image to per-instance (detections)
    # self._flatten_dataset_dicts()

  def _load_detections_from_path(self):
    """
    Detection file: BOP challenge 2023 default detection for task 1
    [{
      "scene_id": 1,
      "image_id": 1,
      "category_id": 29,
      "score": 0.99169921875,
      "bbox": [
        352.40625,
        115.1015625,
        172.125,
        165.5859375
      ],
      "time": 0.16953814402222633
    }, ...]
    """
    score_thres = self.det_thres
    top_k_per_obj = self.det_topk_per_obj

    print(f"Loading detections for {self.name} from: {self.detection_path}")
    # load detections
    detections = load_json(self.detection_path)
    detections_dict = {}
    for det in detections:
      scene_id = det["scene_id"]
      image_id = det["image_id"]
      scene_image_id = f"{scene_id}/{image_id}"
      if scene_image_id not in detections_dict.keys():
        detections_dict[scene_image_id] = []

      detections_dict[scene_image_id].append(det)

    dataset_dicts = self.dataset_dicts

    new_dataset_dicts = []
    for i, record_ori in enumerate(dataset_dicts):
      record = copy.deepcopy(record_ori)
      scene_image_id = record["scene_image_id"]
      if scene_image_id not in detections_dict.keys():
        print(f"No detections found for {scene_image_id}")
        continue

      scene_id = record["scene_id"]

      dets_i = detections_dict[scene_image_id]

      obj_annotations = {obj_id: [] for obj_id in self.objid2catid.keys()}
      for det in dets_i:
        obj_id = det["category_id"]
        bbox_est = det["bbox"]
        time = det.get("time", 0.0)
        score = det.get("score", 1.0)
        if score < score_thres:
          continue
        cat_id = self.objid2catid[obj_id]
        inst = {
          "category_id": cat_id,
          "object_id": obj_id,
          "bbox_est": bbox_est,
          "bbox_mode": "xywh",
          "score": score,
          "time": time,
        }

        if "segmentation" in det.keys():
          # segmentation should e in RLE format either compact or not
          # {'count': ..., 'size': ...}
          inst["segmentation"] = det["segmentation"]

        model_info = self.models_info[obj_id]
        inst["model_info"] = model_info

        obj_annotations[obj_id].append(inst)

      # only select top k detections per object
      annotations = []
      for obj_id, annos in obj_annotations.items():
        scores = [anno["score"] for anno in annos]
        scores_and_annos = sorted(zip(scores, annos), key=lambda x: x[0], reverse=True)
        sel_annos = [anno for _, anno in scores_and_annos][:top_k_per_obj]
        annotations.extend(sel_annos)

      if len(annotations) == 0:
        continue
      record["annotations"] = annotations
      new_dataset_dicts.append(record)

    if len(new_dataset_dicts) < len(dataset_dicts):
      print(
        f"No detections found in {len(dataset_dicts) - len(new_dataset_dicts)} images, original: {len(dataset_dicts)} imgs, left: {len(new_dataset_dicts)} imgs"
      )

    self.dataset_dicts = new_dataset_dicts

  def reduce_categories(self):
    objid2catid = {}  # to 0-base
    catid2objid = {}

    cat_id = 0
    for model_info in self.models_info.values():
      obj_id = model_info["object_id"]
      if obj_id not in self.include_ids:
        continue
      objid2catid[obj_id] = cat_id
      catid2objid[cat_id] = obj_id
      cat_id += 1

    self.objid2catid = objid2catid
    self.catid2objid = catid2objid
    self.num_classes = cat_id

    new_dataset_dicts = []
    for dataset_dict in self.dataset_dicts:
      annotations = []
      for anno in dataset_dict["annotations"]:
        obj_id = anno["object_id"]
        if obj_id not in self.include_ids:
          continue
        cat_id = objid2catid[obj_id]
        anno["category_id"] = cat_id
        annotations.append(anno)
      if len(annotations) == 0:
        continue
      dataset_dict["annotations"] = annotations
      new_dataset_dicts.append(dataset_dict)

    self.dataset_dicts = new_dataset_dicts

  def flatten_annotations(self):
    # refer to detectron2.get_detection_dataset_dicts
    new_dataset_dicts = []
    for dataset_dict in self.dataset_dicts:
      img_infos = {k: v for k, v in dataset_dict.items() if k not in ["annotations"]}
      if "annotations" in dataset_dict:
        for inst_id, anno in enumerate(dataset_dict["annotations"]):
          rec = {"inst_id": inst_id, "inst_infos": anno}
          rec.update(img_infos)
          new_dataset_dicts.append(rec)
      else:
        rec = img_infos
        new_dataset_dicts.append(rec)

    self.dataset_dicts = new_dataset_dicts

  def print_tabulate(self):
    num_classes = len(self.models_info)
    histogram = {obj_id: 0 for obj_id in self.models_info.keys()}
    for dataset_dict in self.dataset_dicts:
      if "annotations" in dataset_dict:
        annos = dataset_dict["annotations"]
        for anno in annos:
          histogram[anno["object_id"]] += 1
      elif "inst_infos" in dataset_dict:
        anno = dataset_dict["inst_infos"]
        histogram[anno["object_id"]] += 1
      else:
        raise ValueError("Neither 'annotations' nor 'inst_infos' in dataset_dict")

    n_cols = min(6, num_classes * 2)

    def short_name(obj_id):
      return self.short_names[obj_id]

    data = list(itertools.chain(*[[short_name(i), v] for i, v in histogram.items()]))
    total_num_instances = sum(data[1::2])
    data.extend([None] * (n_cols - (len(data) % n_cols)))
    if num_classes > 1:
      data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::n_cols] for i in range(n_cols)])
    table = tabulate(
      data,
      headers=["category", "#instances"] * (n_cols // 2),
      tablefmt="pipe",
      numalign="left",
      stralign="center",
    )

    print(table)
