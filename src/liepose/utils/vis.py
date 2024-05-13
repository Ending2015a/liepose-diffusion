import os

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxlie import SO3


def visualize_line(x, title=None, save_path=None, dpi=100):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(x)
  if title is not None:
    ax.set_title(str(title))
  if save_path is not None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, facecolor="w", dpi=dpi)
  plt.close("all")


def circular_shift(x, shift):
  # x in [-pi, pi]
  return (x + np.pi + shift) % (2 * np.pi) - np.pi


# Function adapted from
# https://colab.research.google.com/github/implicit-pdf/implicit-pdf.github.io/blob/main/ipdf_files/ipdf_inference_demo_pascal.ipynb
def visualize_so3_probabilities(
  rotations,
  probabilities=None,
  rotations_gt=None,
  gt_size=2500,
  ax=None,
  fig=None,
  title=None,
  show_ticks=False,
  show_color_wheel=True,
  canonical_rotation=np.eye(3),
  save_path=None,
  dpi=100,
  tight_layout=False,
):
  """Plot a single distribution on SO(3) using the tilt-colored method.

  Args:
    rotations: [N, 3, 3] tensor of rotation matrices
    probabilities: [N] tensor of probabilities
    rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
    ax: The matplotlib.pyplot.axis object to paint
    fig: The matplotlib.pyplot.figure object to paint
    show_color_wheel: If True, display the explanatory color wheel which matches
      color on the plot with tilt angle
    canonical_rotation: A [3, 3] rotation matrix representing the 'display
      rotation', to change the view of the distribution.  It rotates the
      canonical axes so that the view of SO(3) on the plot is different, which
      can help obtain a more informative view.
  Returns:
    A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
  """

  def _show_single_marker(ax, rotation, marker, edgecolors=True, facecolors=False):
    eulers = SO3.from_matrix(rotation).as_rpy_radians()
    xyz = rotation[:, 0]
    tilt_angle = eulers[0]
    longitude = np.arctan2(xyz[0], -xyz[1])
    latitude = np.arcsin(xyz[2])

    color = cmap(0.5 + tilt_angle / 2 / np.pi)
    ax.scatter(
      longitude,
      latitude,
      s=gt_size,
      edgecolors=color if edgecolors else "none",
      facecolors=facecolors if facecolors else "none",
      marker=marker,
      linewidth=4,
    )

  if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="mollweide")
  if rotations_gt is not None and len(rotations_gt.shape) == 2:
    rotations_gt = jnp.expand_dims(rotations_gt, axis=0)

  display_rotations = rotations @ canonical_rotation
  cmap = plt.cm.hsv
  scatterpoint_scaling = 4e3
  eulers_queries = jax.vmap(lambda x: SO3.from_matrix(x).as_rpy_radians())(
    display_rotations
  )
  xyz = display_rotations[:, :, 0]
  tilt_angles = np.array(eulers_queries.roll)

  longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
  latitudes = np.arcsin(xyz[:, 2])

  if not np.all(np.isfinite(longitudes)):
    print("Contains invalid numbers in longitudes")
  if not np.all(np.isfinite(latitudes)):
    print("Contains invalid numbers in latitudes")
  if not np.all(np.isfinite(tilt_angles)):
    print("Contains invalid numbers in tilt_angles")

  mask = np.logical_and(np.isfinite(longitudes), np.isfinite(latitudes))
  mask = np.logical_and(np.isfinite(tilt_angles), mask)
  longitudes = longitudes[mask]
  latitudes = latitudes[mask]
  tilt_angles = tilt_angles[mask]

  #   which_to_display = (probabilities > display_threshold_probability)

  if rotations_gt is not None:
    # The visualization is more comprehensible if the GT
    # rotation markers are behind the output with white filling the interior.
    display_rotations_gt = rotations_gt @ canonical_rotation

    for rotation in display_rotations_gt:
      _show_single_marker(ax, rotation, "o")
    # Cover up the centers with white markers
    for rotation in display_rotations_gt:
      _show_single_marker(ax, rotation, "o", edgecolors=False, facecolors="#ffffff")

  # Display the distribution
  if probabilities is None:
    probabilities = jnp.ones_like(longitudes) * 0.001

  ax.scatter(
    longitudes,
    latitudes,
    s=scatterpoint_scaling * probabilities,
    c=cmap(0.5 + tilt_angles / 2.0 / np.pi),
  )

  ax.grid()
  if not show_ticks:
    ax.set_xticklabels([])
    ax.set_yticklabels([])

  if title is not None:
    ax.set_title(str(title))

  if tight_layout:
    fig.tight_layout(pad=0.00)

  if show_color_wheel:
    # Add a color wheel showing the tilt angle to color conversion.
    ax = fig.add_axes([0.83, 0.16, 0.12, 0.12], projection="polar")
    theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
    radii = np.linspace(0.4, 0.5, 2)
    _, theta_grid = np.meshgrid(radii, theta)
    colormap_val = 0.5 + theta_grid / np.pi / 2.0
    ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
    ax.set_yticklabels([])
    ax.set_xticklabels(
      [
        r"90$\degree$",
        None,
        r"180$\degree$",
        None,
        r"270$\degree$",
        None,
        r"0$\degree$",
      ],
      fontsize=14,
    )
    ax.spines["polar"].set_visible(False)
    plt.text(
      0.5,
      0.5,
      "Tilt",
      fontsize=14,
      horizontalalignment="center",
      verticalalignment="center",
      transform=ax.transAxes,
    )
  if save_path is not None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if tight_layout:
      plt.savefig(save_path, facecolor="w", dpi=dpi, pad_inches=0, bbox_inches="tight")
    else:
      plt.savefig(save_path, facecolor="w", dpi=dpi)


def visualize_r3_probabilities(
  translations,
  probabilities=None,
  translations_gt=None,
  gt_size=2500,
  ax=None,
  fig=None,
  title=None,
  show_ticks=False,
  fixed_range=True,
  save_path=None,
  dpi=100,
  tight_layout=False,
):
  def _show_single_marker(ax, translation, marker, edgecolors=True, facecolors=False):
    x = translation[0]
    y = translation[1]
    z = translation[2]

    color = np.clip((translation + 1.0) / 2.0, 0.0, 1.0)

    ax.scatter(
      x,
      y,
      z,
      s=gt_size,
      edgecolors=[color] if edgecolors else "none",
      facecolors=facecolors if facecolors else "none",
      marker=marker,
      linewidth=4,
    )

  if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
  if translations_gt is not None and len(translations_gt.shape) == 1:
    translations_gt = np.expand_dims(translations_gt, axis=0)

  display_translations = translations
  cmap = plt.cm.hsv
  scatterpoint_scaling = 4e3
  x = display_translations[..., 0]
  y = display_translations[..., 1]
  z = display_translations[..., 2]

  if not np.all(np.isfinite(x)):
    print("Contains invalid numbers in x")
  if not np.all(np.isfinite(y)):
    print("Contains invalid numbers in y")
  if not np.all(np.isfinite(z)):
    print("Contains invalid numbers in z")

  mask = np.logical_and(np.isfinite(x), np.isfinite(y))
  mask = np.logical_and(np.isfinite(z), mask)
  x = x[mask]
  y = y[mask]
  z = z[mask]

  if translations_gt is not None:
    # The visualization is more comprehensible if the GT
    # rotation markers are behind the output with white filling the interior.
    display_translations_gt = translations_gt

    for translation in display_translations_gt:
      _show_single_marker(ax, translation, "o")
    # Cover up the centers with white markers
    for translation in display_translations_gt:
      _show_single_marker(ax, translation, "o", edgecolors=False, facecolors="#ffffff")

  # Display the distribution
  if probabilities is None:
    probabilities = jnp.ones_like(x) * 0.001

  color = np.clip((np.stack([x, y, z], axis=-1) + 1.0) / 2.0, 0.0, 1.0)

  ax.scatter(x, y, z, s=scatterpoint_scaling * probabilities, c=color, zorder=2)

  ax.grid()
  if fixed_range:
    ax.set_xlim([-1.4, 1.4])
    ax.set_ylim([-1.4, 1.4])
    ax.set_zlim([-1.4, 1.4])
    plot_range = np.linspace(-1.0, 1.0, 5, endpoint=True)
    ax.set_xticks(plot_range, plot_range)
    ax.set_yticks(plot_range, plot_range)
    ax.set_zticks(plot_range, plot_range)

  if not show_ticks:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

  ax.set_xlabel("X", labelpad=-12)
  ax.set_ylabel("Y", labelpad=-12)
  ax.set_zlabel("Z", labelpad=-12)

  if title is not None:
    ax.set_title(str(title))

  if save_path is not None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if tight_layout:
      plt.savefig(save_path, facecolor="w", dpi=dpi, pad_inches=0, bbox_inches="tight")
    else:
      plt.savefig(save_path, facecolor="w", dpi=dpi)


def generate_field(n_longitudes: int = 50, n_latitudes: int = 50):
  yaw = np.linspace(-np.pi, np.pi, n_longitudes)
  pitch = np.linspace(-np.pi / 2, np.pi / 2, n_latitudes)
  # matching coordinates
  yaw = circular_shift(yaw, -np.pi / 2)
  pitch = -pitch
  roll = 0

  grid = []
  rots = []
  for i in pitch:
    for j in yaw:
      grid.append([j, i])
      rots.append(SO3.from_rpy_radians(roll, i, j).log())

  grid = np.stack(grid, axis=0)
  rots = jnp.stack(rots, axis=0)
  return grid, rots


def visualize_so3_score_field(
  rotations,
  mesh,
  probabilities=None,
  rotations_marker=None,
  ax=None,
  fig=None,
  show_ticks=False,
  canonical_rotation=np.eye(3),
  save_path=None,
  dpi=100,
):
  """Plot a single distribution on SO(3) using the tilt-colored method.

  Args:
    rotations: [N, 3, 3] tensor of rotation matrices
    probabilities: [N] tensor of probabilities
    rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
    ax: The matplotlib.pyplot.axis object to paint
    fig: The matplotlib.pyplot.figure object to paint
    show_color_wheel: If True, display the explanatory color wheel which matches
      color on the plot with tilt angle
    canonical_rotation: A [3, 3] rotation matrix representing the 'display
      rotation', to change the view of the distribution.  It rotates the
      canonical axes so that the view of SO(3) on the plot is different, which
      can help obtain a more informative view.
  Returns:
    A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
  """
  if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="mollweide")

  display_rotations = rotations @ canonical_rotation
  cmap = plt.cm.hsv
  scatterpoint_scaling = 4e3
  eulers_queries = jax.vmap(lambda x: SO3.from_matrix(x).as_rpy_radians())(
    display_rotations
  )
  # tilt_angles = eulers_queries.roll
  longitudes = circular_shift(eulers_queries.yaw, np.pi / 2)
  latitudes = np.array(-eulers_queries.pitch)

  def generate_palette():
    from matplotlib.colors import Normalize

    ph = np.linspace(0, 2 * np.pi, 13)
    u = np.cos(ph)
    v = np.sin(ph)
    col = np.arctan2(v, u)
    norm = Normalize()
    norm.autoscale(col)
    return norm

  norm = generate_palette()
  th = np.arctan2(latitudes, longitudes)

  ax.quiver(
    mesh[:, 0],
    mesh[:, 1],
    longitudes,
    latitudes,
    color=plt.cm.hsv(norm(th)),
    width=0.0015,
  )

  if rotations_marker is not None:
    display_rotations = rotations_marker @ canonical_rotation
    cmap = plt.cm.hsv
    scatterpoint_scaling = 500
    eulers_queries = jax.vmap(lambda x: SO3.from_matrix(x).as_rpy_radians())(
      display_rotations
    )
    # tilt_angles = eulers_queries.roll
    longitudes = circular_shift(eulers_queries.yaw, np.pi / 2)
    latitudes = np.array(-eulers_queries.pitch)

    ax.scatter(
      longitudes,
      latitudes,
      s=scatterpoint_scaling * probabilities,
      c="black",
      alpha=0.3,
    )
    # c=cmap(0.5 + tilt_angles / 2. / np.pi))

  ax.grid()
  if not show_ticks:
    ax.set_xticklabels([])
    ax.set_yticklabels([])

  if save_path is not None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor="w", dpi=dpi)


def visualize_so3_density(
  rotations, nside, cmap="viridis", save_path=None, dpi=100, title=None
):
  xyz = rotations[:, :, 0]
  phi = jnp.arctan2(xyz[:, 0], -xyz[:, 1])
  theta = jnp.pi / 2 - jnp.arcsin(xyz[:, 2])
  npix = hp.nside2npix(nside)
  theta = jnp.clip(theta, 0, jnp.pi)
  phi = jnp.clip(-phi, -jnp.pi, jnp.pi)
  theta = np.array(theta)
  phi = np.array(phi)

  if not np.all(np.isfinite(theta)):
    print("Contains invalid numbers in theta")
  if not np.all(np.isfinite(phi)):
    print("Contains invalid numbers in phi")

  mask = np.logical_and(np.isfinite(theta), np.isfinite(phi))
  theta = theta[mask]
  phi = phi[mask]

  # convert to HEALPix indices
  indices = hp.ang2pix(nside, theta, phi)
  idx, counts = np.unique(indices, return_counts=True)
  hpx_map = np.zeros(npix, dtype=int)
  hpx_map[idx] = counts

  title = "" if title is None else str(title)

  hp.mollview(hpx_map, cmap=cmap, title=title, cbar=True)
  hp.graticule(dpar=15, dmer=30)

  if save_path is not None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor="w", dpi=dpi)
