from typing import Mapping
import jax.numpy as jnp
from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState


def _alive_indices(world: WorldState) -> jnp.ndarray:
    return jnp.where(world.alive_mask)[0].astype(jnp.int32)


def termination_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _alive_indices(world)
    if idx.size == 0:
        return world
    pos_store = world._get_store("Position")
    term_store = world._get_store("Termination")
    pos = pos_store.read(idx)
    done = jnp.linalg.norm(pos, axis=1) > 10.0
    term_store = term_store.write(idx, done)
    return world._with_store("Termination", term_store)
