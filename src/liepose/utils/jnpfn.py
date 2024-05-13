import jax.numpy as jnp


def normalize(vec, eps=1e-6):
  return vec / jnp.linalg.norm(vec + eps, axis=-1, keepdims=True)


def get_allo_rotation(tran: jnp.ndarray, cam: jnp.ndarray = None, eps: float = 1e-5):
  """Calculate allocentric rotations from the given target translation

  Args:
    tran (jnp.ndarray): _description_
    cam (jnp.array, optional): _description_. Defaults to jnp.float32).

  Return:
    jnp.ndarray: allocentric rotations in quaternion format (wxyz)
  """
  if cam is None:
    cam = jnp.array([0, 0, 1], dtype=jnp.float32)
  cam_ray = normalize(jnp.array(cam), eps=eps)
  # (..., 3)
  obj_ray = normalize(tran, eps=eps)

  ang = jnp.arccos(jnp.clip((cam_ray * obj_ray).sum(axis=-1), -1, 1))  # (...,)
  axis = jnp.cross(cam_ray, obj_ray)  # (..., 3)
  axis = normalize(axis, eps=eps)

  sin_ang = jnp.sin(ang / 2.0)
  cos_ang = jnp.cos(ang / 2.0)

  quat = jnp.stack(
    [
      cos_ang,
      -axis[..., 0] * sin_ang,
      -axis[..., 1] * sin_ang,
      -axis[..., 2] * sin_ang,
    ],
    axis=-1,
  )

  return quat


def reg_loss(ta, out, loss_type="l2"):
  if loss_type == "l2":
    return jnp.mean((ta - out) ** 2)
  elif loss_type == "l1":
    return jnp.mean(jnp.abs(ta - out))
  raise NotImplementedError(f"Unknown loss type: {loss_type}")


def weighted_reg_loss(ta, out, w, loss_type="l2"):
  if loss_type == "l2":
    err = ((ta - out) ** 2) * w
  elif loss_type == "l1":
    err = jnp.abs(ta - out) * w
  else:
    raise NotImplementedError(f"Unknown loss type: {loss_type}")
  err = err * w
  b = err.shape[0]
  err = jnp.sum(err.reshape((b, -1)), axis=-1)
  w = jnp.sum(w.reshape((b, -1)), axis=-1) + 1.0
  return jnp.mean(err / w)
