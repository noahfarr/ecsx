# SPDX-License-Identifier: MIT
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple
import jax
import jax.numpy as jnp

Array = jax.Array
Params = List[Tuple[Array, Array]]  # [(W, b), ...]

def glorot(key, fan_in, fan_out):
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, (fan_in, fan_out), minval=-lim, maxval=+lim)

def init_mlp(key, input_dim: int, hidden: Tuple[int, ...], output_dim: int, activation=jax.nn.tanh) -> Params:
    keys = jax.random.split(key, num=len(hidden) + 1)
    dims = (input_dim, *hidden, output_dim)
    params: Params = []
    for i in range(len(dims) - 1):
        W = glorot(keys[i], dims[i], dims[i + 1])
        b = jnp.zeros((dims[i + 1],), jnp.float32)
        params.append((W, b))
    return params

def apply_mlp(params: Params, x: Array, activation=jax.nn.tanh) -> Array:
    for (W, b) in params[:-1]:
        x = activation(x @ W + b)
    W, b = params[-1]
    return x @ W + b

