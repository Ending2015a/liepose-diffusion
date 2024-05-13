import argparse
import copy
import os
from typing import Tuple

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import open3d as o3d
from jaxlie import SO3
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from liepose.data.symsolt.presets import BasePreset, preset_dict, preset_keys
from liepose.data.utils import load_file, write_file
from liepose.utils import AttrDict
from liepose.utils.jnpfn import get_allo_rotation

ANNOTATION_EXT = ".gz"


class Synthesizer:
  def __init__(
    self,
    width: int = 640,
    height: int = 640,
    hfov: float = 60,
    aspect: float = 1.0,
    x: float = 3,
    visible: bool = False,
    bg_color: Tuple[int, int, int] = [0, 0, 0],
  ):
    a = AttrDict()
    a.width = width
    a.height = height
    a.hfov = np.radians(hfov)
    a.aspect = aspect
    a.x = x
    a.z = a.x / np.tan(a.hfov / 2)
    a.cx = a.width / 2
    a.cy = a.height / 2
    a.fx = (a.width - a.cx) * a.z / a.x
    a.fy = a.fx / a.aspect
    a.bg_color = np.array(bg_color, dtype=np.float32)

    self.a = a
    self.viewer = None

    self._setup_viewer(visible)
    self._setup_camera()

  def _setup_viewer(self, visible=True):
    a = self.a
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(width=a.width, height=a.height, visible=visible)

    opt = viewer.get_render_option()
    opt.background_color = a.bg_color
    opt.light_on = True
    opt.mesh_show_wireframe = False

    self.viewer = viewer

  def _setup_camera(self):
    a = self.a
    cam_param = o3d.camera.PinholeCameraParameters()
    cam_param.extrinsic = np.array(
      [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, a.z], [0, 0, 0, 1]]
    )
    cam_param.intrinsic.set_intrinsics(
      width=a.width, height=a.height, fx=a.fx, fy=a.fy, cx=a.cx, cy=a.cy
    )

    if self.viewer is not None:
      ctr = self.viewer.get_view_control()
      ctr.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)
    return cam_param

  def _sample_poses(
    self,
    num_samples: int,
    scale: float = 1.0,
    max_value: float = None,
    dist_type: str = "uniform",
    seed: int = 40,
  ):
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed), 2)
    if dist_type == "uniform":
      rots = jax.vmap(lambda s: SO3.sample_uniform(s).wxyz)(
        jax.random.split(key1, num_samples)
      )
      trans = jax.random.uniform(key2, shape=(num_samples, 3))
      trans = trans * 2 - 1  # [-1, 1]
    elif dist_type == "sign":
      rots = jax.vmap(lambda s: SO3.sample_uniform(s).wxyz)(
        jax.random.split(key1, num_samples)
      )
      trans = jax.random.uniform(key2, shape=(num_samples, 3))
      trans = trans * 2 - 1  # [-1, 1]
      trans = jnp.sign(trans)
    elif dist_type == "normal":
      rots = jax.vmap(lambda s: SO3.sample_uniform(s).wxyz)(
        jax.random.split(key1, num_samples)
      )
      trans = jax.random.normal(key2, shape=(num_samples, 3))
    elif dist_type == "zero":
      rots = jax.vmap(lambda s: SO3.sample_uniform(s).wxyz)(
        jax.random.split(key1, num_samples)
      )
      trans = jnp.zeros((num_samples, 3), dtype=np.float32)
    elif dist_type == "perspect-demo":
      # sample rotations uniform
      rots = jax.vmap(lambda s: SO3.sample_uniform(s).wxyz)(
        jax.random.split(key1, num_samples // 2 + 1)
      )
      # sample edge translations
      trans = jax.random.uniform(key2, shape=(num_samples // 2 + 1, 3))
      trans = trans * 2 - 1  # [-1, 1]
      trans = jnp.sign(trans)
      # calcualte allocentric angles
      _trans = trans + jnp.array([0, 0, self.a.z])
      cam = jnp.array([0, 0, self.a.z])
      Q = get_allo_rotation(_trans, cam)
      allo_rots = jax.vmap(lambda q, r: (SO3(q) @ SO3(r)).wxyz)(Q, rots)
      allo_trans = jax.vmap(lambda q, t: (SO3(q) @ t))(Q, _trans)
      allo_trans = allo_trans - jnp.array([0, 0, self.a.z])
      rots = jnp.stack((rots, allo_rots), axis=1).reshape((-1, 4))
      trans = jnp.stack((trans, allo_trans), axis=1).reshape((-1, 3))
      rots = rots[:num_samples]
      trans = trans[:num_samples]

    trans = trans * scale
    trans = np.array(trans, dtype=np.float64)
    if max_value is not None:
      trans = np.clip(trans, -max_value, max_value)
    return (np.array(rots, dtype=np.float64), np.array(trans, dtype=np.float64))

  def render_mult(self, meshes, poses):
    self.viewer.clear_geometries()
    for mesh, pose in zip(meshes, poses):
      mesh_posed = copy.deepcopy(mesh)
      mesh_posed.transform(pose)
      self.viewer.add_geometry(mesh_posed)
    self._setup_camera()
    screen_buf = np.asarray(self.viewer.capture_screen_float_buffer(True))
    screen_buf = (screen_buf * 255).astype(np.uint8)
    return screen_buf

  def render(self, mesh, pose):
    mesh_posed = copy.deepcopy(mesh)
    mesh_posed.transform(pose)
    self.viewer.clear_geometries()
    self.viewer.add_geometry(mesh_posed)
    self._setup_camera()
    screen_buf = np.asarray(self.viewer.capture_screen_float_buffer(True))
    screen_buf = (screen_buf * 255).astype(np.uint8)
    return screen_buf

  def synthesize(
    self,
    mesh,
    mesh_color,
    preset: BasePreset,
    path: str,
    translation_scale: float = 1.0,
    num_samples: int = 100,
    num_points: int = 1024,
    hard_annotated: bool = False,
    dist_type: str = "uniform",
    seed: int = 40,
  ):
    # create directories
    root_path = path
    image_dir = os.path.join(root_path, "images")
    os.makedirs(image_dir, exist_ok=True)

    annotations = []

    rots, trans = self._sample_poses(
      num_samples, scale=translation_scale, dist_type=dist_type, seed=seed
    )
    rots = np.roll(rots, -1, axis=-1)
    rots = Rotation.from_quat(rots).as_matrix()

    gt_poses = np.stack([np.eye(4)] * num_samples, axis=0)
    gt_poses[:, :3, :3] = rots
    gt_poses[:, :3, 3] = trans

    for index in tqdm(range(num_samples)):
      # random sample pose
      gt_pose = gt_poses[index]
      # eq_poses = preset.get_symmetric_poses(gt_pose)

      # render image
      screen_buf = self.render(mesh, gt_pose)

      filename = f"{index:06d}.png"

      imageio.imwrite(os.path.join(image_dir, filename), screen_buf)

      eq_poses = []
      if preset.is_ambiguous(screen_buf):
        eq_poses = preset.get_symmetric_poses(gt_pose)
        if hard_annotated:
          # find all ambiguous poses
          ambig_poses = []
          # TODO: dirty fix

          if len(eq_poses) == 720:
            visible_mask = [True] * 361
            prev_idx = 0
            for pose_idx in range(0, 361, 15):
              screen_buf = self.render(mesh, eq_poses[pose_idx])
              if preset.is_ambiguous(screen_buf):
                visible_mask[pose_idx] = False
                if visible_mask[prev_idx] is False:
                  visible_mask
                  visible_mask[prev_idx:pose_idx] = [False] * (pose_idx - prev_idx)
                if visible_mask[prev_idx] is True:
                  for idx in range(prev_idx, pose_idx):
                    screen_buf = self.render(mesh, eq_poses[idx])
                    if preset.is_ambiguous(screen_buf):
                      visible_mask[idx] = False
              else:
                if visible_mask[prev_idx] is False:
                  for idx in range(prev_idx, pose_idx):
                    screen_buf = self.render(mesh, eq_poses[idx])
                    if preset.is_ambiguous(screen_buf):
                      visible_mask[idx] = False
              prev_idx = pose_idx
            visible_mask = visible_mask[:-1] + visible_mask[:-1]

            for pose_idx, visible in enumerate(visible_mask):
              if visible is False:
                ambig_poses.append(eq_poses[pose_idx])
            eq_poses = ambig_poses

          else:
            for pose in eq_poses:
              screen_buf = self.render(mesh, pose)
              if preset.is_ambiguous(screen_buf):
                ambig_poses.append(pose)
          eq_poses = ambig_poses

      if len(eq_poses) == 0:
        eq_poses = [gt_pose]

      eq_poses = np.array(eq_poses)
      num_sym_poses = len(eq_poses)

      # render image (colored)
      screen_buf = self.render(mesh_color, gt_pose)

      filename_color = f"{index:06d}_color.png"

      imageio.imwrite(os.path.join(image_dir, filename_color), screen_buf)

      # convert matrix to quaternion
      gt_rot = Rotation.from_matrix(gt_pose[:3, :3]).as_quat()
      gt_rot = np.roll(gt_rot, 1, axis=-1)  # xyzw -> wxyz
      eq_rots = Rotation.from_matrix(eq_poses[:, :3, :3]).as_quat()
      eq_rots = np.roll(eq_rots, 1, axis=-1)  # xyzw -> wxyz
      gt_tran = gt_pose[:3, 3]

      anno = {
        "image_id": index,
        "image": os.path.join("images", filename),
        "image_color": os.path.join("images", filename_color),
        "points": "points.ply",
        "rotation": gt_rot.ravel().tolist(),  # (4,)
        "rotations_equivalent": eq_rots.ravel().tolist(),  # (n*4,)
        "translation": gt_tran.ravel().tolist(),  # (3,)
        "num_equivalents": num_sym_poses,
        "label_shape": preset.obj_index,
      }
      annotations.append(anno)

    anno_path = os.path.join(root_path, "annotations" + ANNOTATION_EXT)
    write_file(anno_path, annotations)

    # sample point clouds
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    o3d.io.write_point_cloud(
      os.path.join(root_path, "points.ply"),
      pcd,
    )
    return anno_path

  def view(self):
    self._setup_camera()
    self.viewer.run()
    self.viewer.destroy_window()


def merge_annotations(anno_list, shape_names):
  annotations = []
  assert len(anno_list) == len(shape_names)
  for anno_path, shape_name in zip(anno_list, shape_names):
    annos = load_file(anno_path)

    for anno in annos:
      anno["image"] = os.path.join(shape_name, anno["image"])
      anno["image_color"] = os.path.join(shape_name, anno["image_color"])
      anno["points"] = os.path.join(shape_name, anno["points"])

      annotations.append(anno)
  return annotations


def get_args():
  parser = argparse.ArgumentParser(description="SymsolT Dataset")
  parser.add_argument("--path", type=str, default="./dataset/symsolt/test-5k-v2")
  parser.add_argument(
    "--config", type=str, default=None, help="restore arguments from YAML"
  )
  parser.add_argument("dot_list", nargs=argparse.REMAINDER)
  return parser.parse_args()


def load_from_file(config):
  a = OmegaConf.load(config)
  OmegaConf.resolve(a)
  return a


def load_config():
  a = AttrDict()

  a.num_samples = 25000  # = num samples per shape * num shapes
  a.shape_names = preset_keys
  a.seed = 82973
  a.num_points = 1024  # point cloud
  a.hard_annotated = False
  a.dist_type = "uniform"

  a.width = 224
  a.height = 224
  a.hfov = 60
  a.aspect = 1.0
  a.x = 5.25
  a.shape_size = 3.5
  a.max_translation = 1.0

  return OmegaConf.create(dict(a))


def main():
  a = get_args()
  path = a.path

  # load config
  conf = load_config()
  if a.config is not None:
    conf = OmegaConf.merge(conf, OmegaConf.load(a.config))
  if a.dot_list is not None:
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(a.dot_list))
  OmegaConf.resolve(conf)
  # save config
  os.makedirs(path, exist_ok=True)
  config_path = os.path.join(path, "config.yaml")
  OmegaConf.save(config=conf, f=config_path)

  a = OmegaConf.to_container(conf, resolve=True)
  a = AttrDict(a)

  s = Synthesizer(
    width=a.width,
    height=a.height,
    hfov=a.hfov,
    aspect=a.aspect,
    x=a.x,
  )

  num_samples_per_shape = a.num_samples // len(a.shape_names)
  anno_paths = []
  for shape_name in a.shape_names:
    preset = preset_dict[shape_name]
    mesh = preset.create(size=a.shape_size)
    mesh_color = preset.create(size=a.shape_size, color=True)
    shape_path = os.path.join(path, shape_name)

    anno_path = s.synthesize(
      mesh=mesh,
      mesh_color=mesh_color,
      preset=preset,
      path=shape_path,
      translation_scale=a.max_translation,
      num_samples=num_samples_per_shape,
      num_points=a.num_points,
      hard_annotated=a.hard_annotated,
      dist_type=a.dist_type,
      seed=a.seed + preset.obj_index,
    )
    anno_paths.append(anno_path)

  annotations = merge_annotations(anno_paths, a.shape_names)

  anno_path = os.path.join(path, "annotations" + ANNOTATION_EXT)
  write_file(anno_path, annotations)


if __name__ == "__main__":
  main()
