from typing import Mapping
import jax.numpy as jnp
from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState


def _alive_indices(world: WorldState) -> jnp.ndarray:
    return jnp.where(world.alive_mask)[0].astype(jnp.int32)


def action_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _alive_indices(world)
    if idx.size == 0:
        return world
    pos_store = world._get_store("Position")
    act_store = world._get_store("Action")
    pos = pos_store.read(idx)
    act = act_store.read(idx)
    pos_store = pos_store.write(idx, pos + act)
    act_store = act_store.write(idx, jnp.zeros_like(act))
    world = world._with_store("Position", pos_store)
    world = world._with_store("Action", act_store)
    return world
