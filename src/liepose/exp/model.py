from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_resnet
from flax.core import FrozenDict
from flax.core.frozen_dict import unfreeze

activ_dict = {
  "relu": jax.nn.relu,
  "leaky_relu": jax.nn.leaky_relu,
  "silu": jax.nn.silu,
  "sigmoid": jax.nn.sigmoid,
  "gelu": jax.nn.gelu,
  "softplus": jax.nn.softplus,
  "sin": jnp.sin,
}


def get_activ_fn(activ_name: str):
  activ_name = activ_name.lower()
  if activ_name not in activ_dict.keys():
    raise ValueError(f"Unknown activ fn: {activ_name}")

  return activ_dict[activ_name]


def broadcast_batch(x, bs):
  b = x.shape[0]
  s = bs // b

  x = jnp.expand_dims(x, axis=1)  # (b, 1, dim)
  x = jnp.repeat(x, s, axis=1).reshape((bs, -1))  # (bs, dim)
  return x


class PosEmbed(nn.Module):
  embed_dim: int
  dim: int
  shift: bool = True

  def setup(self):
    half_dim = self.embed_dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    self.emb = emb
    self.mlp = nn.Dense(self.dim, use_bias=self.shift)

  def __call__(self, x):
    x = jnp.expand_dims(x, axis=-1)
    emb = x * self.emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    emb = emb.reshape(emb.shape[:-2] + (-1,))
    emb = self.mlp(emb)
    return emb


class MlpBlock(nn.Module):
  dim: int
  activ: Callable = jax.nn.leaky_relu

  @nn.compact
  def __call__(self, x_in, c):
    activ = self.activ
    c = nn.Dense(self.dim * 2)(jax.nn.silu(c))
    s, b = jnp.split(c, 2, axis=-1)

    x = nn.Dense(self.dim)(activ(x_in))
    x = x * (s + 1) + b
    x = nn.Dense(self.dim)(x)
    return x + nn.Dense(self.dim)(x_in)


class FourierMlpBlock(nn.Module):
  dim: int
  activ: Callable = jnp.sin

  @nn.compact
  def __call__(self, x_in, c):
    activ = self.activ
    c = nn.Dense(self.dim * 2)(jax.nn.silu(c))
    a, b = jnp.split(c, 2, axis=-1)

    x = nn.Dense(self.dim)(activ(x_in))
    x = a * jnp.cos(x * jnp.pi) + b * jnp.sin(x * jnp.pi)
    x = nn.Dense(self.dim)(x)
    return x + nn.Dense(self.dim)(x_in)


class Backbone(nn.Module):
  dim: int
  feat_dim: int
  resnet: nn.Module
  activ: Callable = jax.nn.leaky_relu

  def setup(self):
    self.conv = nn.Conv(self.feat_dim, kernel_size=(1, 1), strides=(1, 1), name="conv")
    self.dense = nn.Dense(self.dim, name="dense")

  def __call__(self, x):
    activ = self.activ
    x, _ = self.resnet(x)
    x = self.conv(x)
    b = x.shape[0]
    x = x.reshape((b, -1))
    x = self.dense(activ(x))
    return x


class Head(nn.Module):
  dim: int
  n_layers: int = 1
  block_type: nn.Module = MlpBlock
  activ: Callable = jax.nn.leaky_relu

  @nn.compact
  def __call__(self, x0, rt, t):
    activ = self.activ
    # x0 (b, 512) -> (bs, 512)
    x0 = broadcast_batch(x0, rt.shape[0])
    t = PosEmbed(256, 256, True)(t)
    x = PosEmbed(256, 256, False)(rt)
    for _ in range(self.n_layers):
      x = self.block_type(256, activ)(self.block_type(256, activ)(x, x0), t)
    return nn.Dense(self.dim)(activ(x))


class Model(nn.Module):
  backbone: nn.Module
  head: nn.Module

  def __call__(self, img, rt, t):
    x = self.backbone(img)
    mu = self.head(x, rt, t)
    return mu


def _create_backbone(
  seed, dim=512, feat_dim=64, image_shape=[1, 224, 224, 3], resnet_depth=50
):
  # load pretrained resnet backbone
  ResNet, params = jax_resnet.pretrained_resnet(resnet_depth)
  resnet, resnet_params = jax_resnet.slice_model_and_variables(
    ResNet(), params, start=0, end=-2
  )
  # create backbone
  backbone = Backbone(dim, feat_dim, resnet)  # leaky relu
  # create initial params
  backbone_params = backbone.init(seed, jnp.ones(image_shape))

  if image_shape[-1] != 3:
    # replace params of the first conv layer
    resnet_params = unfreeze(resnet_params)
    resnet_params["params"]["layers_0"]["ConvBlock_0"]["Conv_0"] = backbone_params[
      "params"
    ]["resnet"]["layers_0"]["ConvBlock_0"]["Conv_0"]
    resnet_params = FrozenDict(resnet_params)
  # replace resnet params with pretrained params
  backbone_params = FrozenDict(
    {
      "params": {
        "resnet": resnet_params["params"],
        "conv": backbone_params["params"]["conv"],
        "dense": backbone_params["params"]["dense"],
      },
      "batch_stats": {"resnet": resnet_params["batch_stats"]},
    }
  )

  return backbone, backbone_params


def create_model_fn(
  out_dim=6,
  in_dim=6,
  image_shape=[1, 224, 224, 3],
  resnet_depth=34,
  mlp_layers=1,
  fourier_block=True,
  activ_fn="leaky_relu",
):
  def create_model_and_params(seed):
    # create backbone model
    backbone, backbone_params = _create_backbone(
      seed, dim=512, feat_dim=128, image_shape=image_shape, resnet_depth=resnet_depth
    )
    # expected: (1, 512)
    backbone_output = backbone.apply(backbone_params, jnp.ones(image_shape))
    # create head
    block_type = FourierMlpBlock if fourier_block else MlpBlock
    head = Head(
      dim=out_dim,
      n_layers=mlp_layers,
      block_type=block_type,
      activ=get_activ_fn(activ_fn),
    )
    head_params = head.init(
      seed,
      jnp.ones(backbone_output.shape),
      jnp.ones([1, in_dim]),  # current pose
      jnp.ones([1, 1]),  # time
    )
    # combine resnet backbone and head
    model = Model(backbone, head)
    params = FrozenDict(
      {
        "params": {
          "backbone": backbone_params["params"],
          "head": head_params["params"],
        },
        "batch_stats": {
          "backbone": backbone_params["batch_stats"],
        },
      }
    )
    return model, params

  return create_model_and_params
