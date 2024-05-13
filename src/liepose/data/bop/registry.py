DATASET_LIST = {}
DATASET_TYPE_LIST = {}


def register_dataset(
  dataset_type,
  dataset_name,
  dataset_path,
  models_info_path,
  detection_path=None,
  **kwargs,
):
  global DATASET_LIST
  assert (
    dataset_name not in DATASET_LIST.keys()
  ), f"The dataset '{dataset_name}' is registered."
  DATASET_LIST[dataset_name] = dict(
    dataset_type=dataset_type,
    path=dataset_path,
    models_info_path=models_info_path,
    detection_path=detection_path,
    **kwargs,
  )


def get_dataset(dataset_name):
  global DATASET_LIST
  assert (
    dataset_name in DATASET_LIST.keys()
  ), f"The dataset '{dataset_name}' is not registered."
  return DATASET_LIST[dataset_name]


def register_dataset_type(dataset_type: str):
  """Register dataset type

  ```
  @register_dataset_type('my-datset-type')
  class MyDataset(Dataset):
    def __init__(self, ...):
      ...
  ```
  """
  global DATASET_TYPE_LIST
  assert (
    dataset_type not in DATASET_TYPE_LIST.keys()
  ), f"The dataset type '{dataset_type}' is registered."

  def _register_dataset_type(dataset_class: type):
    assert (
      dataset_type not in DATASET_TYPE_LIST.keys()
    ), f"The dataset type '{dataset_type}' is registered."
    if (
      not hasattr(dataset_class, "dataset_type")
    ) or dataset_class.dataset_type is None:
      dataset_class.dataset_type = dataset_type
    DATASET_TYPE_LIST[dataset_type] = dataset_class
    return dataset_class

  return _register_dataset_type


def get_dataset_type(dataset_type: str):
  global DATASET_TYPE_LIST
  assert (
    dataset_type in DATASET_TYPE_LIST.keys()
  ), f"The dataset type '{dataset_type}' is not registered."
  return DATASET_TYPE_LIST[dataset_type]
