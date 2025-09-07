from typing import Mapping
import jax.numpy as jnp
from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState


def _entity_indices(world: WorldState) -> jnp.ndarray:
    """Static range of entity indices for use inside ``jax.jit``."""
    return jnp.arange(world.alive_mask.shape[0], dtype=jnp.int32)


def termination_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _entity_indices(world)
    alive = world.alive_mask.astype(bool)
    pos_store = world._get_store("Position")
    term_store = world._get_store("Termination")
    pos = pos_store.read(idx)
    done = jnp.linalg.norm(pos, axis=1) > 10.0
    term = term_store.read(idx)
    term = jnp.where(alive, done, term)
    term_store = term_store.write(idx, term)
    return world._with_store("Termination", term_store)
