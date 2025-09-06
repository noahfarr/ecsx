from typing import Mapping
import jax.numpy as jnp
from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState


def _alive_indices(world: WorldState) -> jnp.ndarray:
    return jnp.where(world.alive_mask)[0].astype(jnp.int32)


def reward_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _alive_indices(world)
    if idx.size == 0:
        return world
    pos_store = world._get_store("Position")
    rew_store = world._get_store("Reward")
    pos = pos_store.read(idx)
    reward = -jnp.linalg.norm(pos, axis=1)
    rew_store = rew_store.write(idx, reward)
    return world._with_store("Reward", rew_store)
