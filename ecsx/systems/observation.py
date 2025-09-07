from typing import Mapping
import jax
import jax.numpy as jnp
from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState


def _entity_indices(world: WorldState) -> jnp.ndarray:
    """Return indices for all entities.

    Similar to ``systems.action``, using ``jnp.where`` with ``alive_mask`` leads
    to arrays with data-dependent shapes that break under ``jax.jit``.  We
    instead operate on the full static index range and mask out inactive
    entities downstream.
    """

    return jnp.arange(world.alive_mask.shape[0], dtype=jnp.int32)


def observation_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _entity_indices(world)
    alive = world.alive_mask.astype(bool)
    pos_store = world._get_store("Position")
    obs_store = world._get_store("Observation")
    pos = pos_store.read(idx)
    obs = obs_store.read(idx)
    obs = jnp.where(alive[:, None], pos, obs)
    obs_store = obs_store.write(idx, obs)
    return world._with_store("Observation", obs_store)


def grid_observation_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _alive_indices(world)
    if idx.size == 0:
        return world
    view_radius = int(inputs.get("view_radius", 0))
    pos_store = world._get_store("Position")
    obs_store = world._get_store("Observation")
    agent_pos = pos_store.read(idx)
    try:
        obst_store = world._get_store("Obstacle")
        obst_idx = jnp.where(obst_store.alive_mask)[0]
        obst_pos = pos_store.read(obst_idx)
    except KeyError:
        obst_pos = jnp.empty((0, agent_pos.shape[1]))
    goal_pos = inputs.get("goal_position")

    def build_obs(a_pos):
        size = 2 * view_radius + 1
        patch = jnp.zeros((size, size), jnp.int32)
        if obst_pos.shape[0] > 0:
            rel = obst_pos - a_pos
            mask = jnp.all(jnp.abs(rel) <= view_radius, axis=1)
            rel = rel[mask].astype(jnp.int32)
            coords = rel + view_radius
            patch = patch.at[coords[:, 0], coords[:, 1]].set(1)
        if goal_pos is not None:
            goal = jnp.asarray(goal_pos, a_pos.dtype)
            relg = goal - a_pos
            cond = jnp.all(jnp.abs(relg) <= view_radius)
            coord = (relg.astype(jnp.int32) + view_radius).clip(0, size - 1)
            patch = jnp.where(cond, patch.at[coord[0], coord[1]].set(2), patch)
        patch = patch.at[view_radius, view_radius].set(3)
        return patch.reshape(-1)

    obs = jax.vmap(build_obs)(agent_pos)
    obs_store = obs_store.write(idx, obs)
    return world._with_store("Observation", obs_store)
