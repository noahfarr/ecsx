from typing import Mapping
import jax.numpy as jnp
from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState


def _entity_indices(world: WorldState) -> jnp.ndarray:
    """Return indices for all entities in the world.

    Using ``jnp.where`` to select active indices produces arrays with
    data-dependent shapes that cannot be used inside ``jax.jit``.  Instead we
    return a static arange of all possible indices and rely on boolean masks to
    ignore inactive entities when updating state.
    """

    return jnp.arange(world.alive_mask.shape[0], dtype=jnp.int32)


def _apply_collision(
    world: WorldState, idx: jnp.ndarray, new_pos: jnp.ndarray, pos: jnp.ndarray
) -> jnp.ndarray:
    obstacle_store = world._get_store("Obstacle")
    position_store = world._get_store("Position")

    obstacle_mask = obstacle_store.alive_mask.astype(bool)
    all_idx = jnp.arange(obstacle_mask.shape[0], dtype=jnp.int32)
    obstacle_pos = position_store.read(all_idx)
    is_obstacle = obstacle_mask[idx]
    same_position = jnp.all(new_pos[:, None, :] == obstacle_pos[None, :, :], axis=-1)
    collision = jnp.any(jnp.logical_and(same_position, obstacle_mask[None, :]), axis=1)
    collision = jnp.logical_and(collision, jnp.logical_not(is_obstacle))
    return jnp.where(collision[:, None], pos, new_pos)


def grid_movement_system(world: WorldState, key: Key, inputs: Mapping[str, Array]):
    idx = _entity_indices(world)
    alive = world.alive_mask.astype(bool)

    position_store = world._get_store("Position")
    action_store = world._get_store("DiscreteAction")

    position = position_store.read(idx)
    action = action_store.read(idx)
    directions = jnp.array(
        [[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]], dtype=jnp.float32
    )
    next_position = position + directions[action]
    next_position = _apply_collision(world, idx, next_position, position)

    next_position = jnp.where(alive, next_position, position)
    next_action = jnp.where(alive, jnp.zeros_like(action), action)

    position_store = position_store.write(idx, next_position)
    action_store = action_store.write(idx, next_action)
    world = world._with_store(
        "Position", world._get_store("Position").write(idx, next_position)
    )
    world = world._with_store(
        "DiscreteAction", world._get_store("DiscreteAction").write(idx, next_action)
    )
