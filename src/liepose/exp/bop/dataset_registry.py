import os

from liepose.data.bop import BOPPoseDataset, register_dataset


def datadir(path):
  return os.path.join(os.environ.get("DATADIR", "./"), path)


##### TLESS #####

register_dataset(
  "tless",
  "tless-train_pbr",
  datadir("bop_datasets/tless/train_pbr/"),
  datadir("bop_datasets/tless/models_cad/models_info.json"),
)

register_dataset(
  "tless",
  "tless-train_primesense",
  datadir("bop_datasets/tless/train_primesense/"),
  datadir("bop_datasets/tless/models_cad/models_info.json"),
)

register_dataset(
  "tless",
  "tless-test_primesense",
  datadir("bop_datasets/tless/test_primesense/"),
  datadir("bop_datasets/tless/models_cad/models_info.json"),
  test_bop_path=datadir("bop_datasets/tless/test_targets_bop19.json"),
  filter_invalid=True,
)

##### ICBIN #####

register_dataset(
  "icbin",
  "icbin-train_pbr",
  datadir("bop_datasets/icbin/train_pbr/"),
  datadir("bop_datasets/icbin/models/models_info.json"),
)

register_dataset(
  "icbin",
  "icbin-train",
  datadir("bop_datasets/icbin/train/"),
  datadir("bop_datasets/icbin/models/models_info.json"),
)

register_dataset(
  "icbin",
  "icbin-test",
  datadir("bop_datasets/icbin/test/"),
  datadir("bop_datasets/icbin/models/models_info.json"),
  test_bop_path=datadir("bop_datasets/icbin/test_targets_bop19.json"),
  filter_invalid=True,
)

if __name__ == "__main__":
  BOPPoseDataset("tless-train_pbr")
  BOPPoseDataset("tless-train_primesense")
  BOPPoseDataset("tless-test_primesense")
  BOPPoseDataset("icbin-train_pbr")
  BOPPoseDataset("icbin-train")
  BOPPoseDataset("icbin-test")
