# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import replace
import jax
import jax.numpy as jnp
from ..core.world_state import WorldState


def split_key(world: WorldState):
    """Split and store back the next PRNG key; return (world, subkey)."""
    key, sub = jax.random.split(world.random_key)
    return replace(world, random_key=key), sub


def uniform(subkey, shape, low=0.0, high=1.0, dtype=jnp.float32):
    return jax.random.uniform(subkey, shape, dtype=dtype) * (high - low) + low


def normal(subkey, shape, dtype=jnp.float32):
    return jax.random.normal(subkey, shape, dtype=dtype)

