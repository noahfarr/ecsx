from typing import Any, Mapping
import jax.numpy as jnp
from ecsx.core.typing import Key
from ecsx.core.world import WorldState

from ecsx.components import Position


def move_system(world: WorldState, key: Key, inputs: Mapping[str, Any]) -> WorldState:
    """Move entities in the world."""
    move_store = world._get_store(inputs["move_store"])
    position_store = world._get_store(inputs["position_store"])

    move = move_store.data
    position = position_store.data
    mask = world.alive_mask & position_store.alive_mask & move_store.alive_mask

    x = jnp.where(mask, position.x + move.vx, position.x)
    y = jnp.where(mask, position.y + move.vy, position.y)

    position_store = position_store.write(world.alive_mask, Position(x=x, y=y))

    world = world._with_store(inputs["position_store"], position_store)

    return world
