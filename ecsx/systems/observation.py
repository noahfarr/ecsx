from typing import Mapping
import jax.numpy as jnp
from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState


def _alive_indices(world: WorldState) -> jnp.ndarray:
    return jnp.where(world.alive_mask)[0].astype(jnp.int32)


def observation_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _alive_indices(world)
    if idx.size == 0:
        return world
    pos_store = world._get_store("Position")
    obs_store = world._get_store("Observation")
    obs_store = obs_store.write(idx, pos_store.read(idx))
    return world._with_store("Observation", obs_store)
