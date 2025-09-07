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
    try:
        obst_store = world._get_store("Obstacle")
    except KeyError:
        return new_pos
    pos_store = world._get_store("Position")
    obstacle_mask = obst_store.alive_mask.astype(bool)
    all_idx = jnp.arange(obstacle_mask.shape[0], dtype=jnp.int32)
    obstacle_pos = pos_store.read(all_idx)
    is_obstacle = obstacle_mask[idx]
    same_position = jnp.all(
        new_pos[:, None, :] == obstacle_pos[None, :, :], axis=-1
    )
    collision = jnp.any(
        jnp.logical_and(same_position, obstacle_mask[None, :]), axis=1
    )
    collision = jnp.logical_and(collision, jnp.logical_not(is_obstacle))
    return jnp.where(collision[:, None], pos, new_pos)


def discrete_action_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _entity_indices(world)
    alive = world.alive_mask.astype(bool)
    pos_store = world._get_store("Position")
    act_store = world._get_store("DiscreteAction")
    pos = pos_store.read(idx)
    act = act_store.read(idx)
    directions = jnp.array(
        [[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]], dtype=jnp.float32
    )
    act = jnp.clip(act, 0, 4)
    delta = directions[act]
    new_pos = pos + delta
    grid_size = inputs.get("grid_size")
    if grid_size is not None:
        max_bounds = jnp.asarray(grid_size, dtype=new_pos.dtype) - 1
        new_pos = jnp.clip(new_pos, 0, max_bounds)
    new_pos = _apply_collision(world, idx, new_pos, pos)
    pos = jnp.where(alive[:, None], new_pos, pos)
    pos_store = pos_store.write(idx, pos)
    reset_act = jnp.where(alive, jnp.zeros_like(act), act)
    act_store = act_store.write(idx, reset_act)
    world = world._with_store("Position", pos_store)
    world = world._with_store("DiscreteAction", act_store)
    return world


def continuous_action_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _entity_indices(world)
    alive = world.alive_mask.astype(bool)
    pos_store = world._get_store("Position")
    act_store = world._get_store("ContinuousAction")
    pos = pos_store.read(idx)
    act = act_store.read(idx)
    new_pos = pos + act
    grid_size = inputs.get("grid_size")
    if grid_size is not None:
        max_bounds = jnp.asarray(grid_size, dtype=new_pos.dtype) - 1
        new_pos = jnp.clip(new_pos, 0, max_bounds)
    new_pos = _apply_collision(world, idx, new_pos, pos)
    pos = jnp.where(alive[:, None], new_pos, pos)
    pos_store = pos_store.write(idx, pos)
    reset_act = jnp.where(alive[:, None], jnp.zeros_like(act), act)
    act_store = act_store.write(idx, reset_act)
    world = world._with_store("Position", pos_store)
    world = world._with_store("ContinuousAction", act_store)
    return world
