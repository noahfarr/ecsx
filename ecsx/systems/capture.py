from typing import Mapping

import jax.numpy as jnp

from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState
from ecsx.core.component_store import ComponentStore


def flag_capture_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    """Handle flag pickup, carrying and scoring."""

    pos_store = world._get_store("Position")
    team_store = world._get_store("Team")
    flag_store = world._get_store("Flag")
    rew_store = world._get_store("Reward")

    idx = jnp.arange(world.alive_mask.shape[0], dtype=jnp.int32)
    team_mask = team_store.alive_mask.astype(bool)
    flag_mask = flag_store.alive_mask.astype(bool)

    agent_pos = pos_store.read(idx)
    agent_team = team_store.read(idx)

    flag_data = flag_store.read(idx)
    flag_pos = pos_store.read(idx)
    owner = flag_data[:, 0]
    carried = flag_data[:, 1].astype(bool)
    carrier_ids = flag_data[:, 2]

    # Capture flags that are not currently carried
    flag_free = jnp.logical_and(flag_mask, jnp.logical_not(carried))
    same_pos = jnp.all(agent_pos[:, None, :] == flag_pos[None, :, :], axis=-1)
    enemy = agent_team[:, None] != owner[None, :]
    capture_mat = jnp.logical_and(same_pos, enemy)
    capture_mat = jnp.logical_and(capture_mat, team_mask[:, None])
    capture_mat = jnp.logical_and(capture_mat, flag_free[None, :])
    any_capture = jnp.any(capture_mat, axis=0)
    captor_idx = jnp.argmax(jnp.where(capture_mat, 1, 0), axis=0)
    carrier_entity = idx[captor_idx]
    carried = jnp.where(flag_free, jnp.where(any_capture, True, carried), carried)
    carrier_ids = jnp.where(
        flag_free, jnp.where(any_capture, carrier_entity, carrier_ids), carrier_ids
    )

    # Move carried flags with their carriers
    carrier_ids_safe = jnp.where(flag_mask, carrier_ids, 0)
    carried_pos = pos_store.read(carrier_ids_safe)
    flag_pos = jnp.where(
        jnp.logical_and(flag_mask, carried)[:, None], carried_pos, flag_pos
    )

    # Check for scoring when carrier reaches its base
    base_positions = inputs.get("base_positions")
    if base_positions is None:
        base_positions = jnp.zeros((2, 2), jnp.int32)
    carrier_team = team_store.read(carrier_ids_safe)
    carrier_base = base_positions[carrier_team]
    scored = jnp.logical_and(
        jnp.logical_and(flag_mask, carried),
        jnp.all(carried_pos == carrier_base, axis=1),
    )
    rew = rew_store.read(idx)
    rew = rew.at[carrier_ids_safe].add(jnp.where(scored, 1.0, 0.0))
    home_pos = base_positions[owner]
    flag_pos = jnp.where(scored[:, None], home_pos, flag_pos)
    carried = jnp.where(scored, False, carried)
    carrier_ids = jnp.where(scored, -1, carrier_ids)

    flag_out = jnp.stack([owner, carried.astype(jnp.int32), carrier_ids], axis=1)
    new_flag_data = jnp.where(flag_mask[:, None], flag_out, flag_store.data)
    flag_store = ComponentStore(flag_store.spec, new_flag_data, flag_store.alive_mask)
    new_pos_data = jnp.where(flag_mask[:, None], flag_pos, pos_store.data)
    pos_store = ComponentStore(pos_store.spec, new_pos_data, pos_store.alive_mask)
    rew_store = rew_store.write(idx, rew)
    world = world._with_store("Flag", flag_store)
    world = world._with_store("Position", pos_store)
    world = world._with_store("Reward", rew_store)
    return world
