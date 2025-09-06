from typing import Mapping
import jax
import jax.numpy as jnp
from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState


def _alive_indices(world: WorldState) -> jnp.ndarray:
    return jnp.where(world.alive_mask)[0].astype(jnp.int32)


def _apply_collision(
    world: WorldState, idx: jnp.ndarray, new_pos: jnp.ndarray, pos: jnp.ndarray
) -> jnp.ndarray:
    try:
        obst_store = world._get_store("Obstacle")
    except KeyError:
        return new_pos
    pos_store = world._get_store("Position")
    obstacle_indices = jnp.where(obst_store.alive_mask)[0]
    if obstacle_indices.size == 0:
        return new_pos
    obstacle_pos = pos_store.read(obstacle_indices)
    is_obstacle = obst_store.read(idx).astype(bool)
    collision = jnp.any(
        jnp.all(new_pos[:, None, :] == obstacle_pos[None, :, :], axis=-1), axis=1
    )
    collision = jnp.logical_and(collision, jnp.logical_not(is_obstacle))
    return jnp.where(collision[:, None], pos, new_pos)


def discrete_action_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _alive_indices(world)
    if idx.size == 0:
        return world
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
    pos_store = pos_store.write(idx, new_pos)
    act_store = act_store.write(idx, jnp.zeros_like(act))
    world = world._with_store("Position", pos_store)
    world = world._with_store("DiscreteAction", act_store)
    return world


def continuous_action_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _alive_indices(world)
    if idx.size == 0:
        return world
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
    pos_store = pos_store.write(idx, new_pos)
    act_store = act_store.write(idx, jnp.zeros_like(act))
    world = world._with_store("Position", pos_store)
    world = world._with_store("ContinuousAction", act_store)
    return world
