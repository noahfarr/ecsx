from typing import Mapping

import jax.numpy as jnp

from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState


def flag_capture_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    """Handle flag pickup, carrying and scoring."""

    # Required component stores
    pos_store = world._get_store("Position")
    team_store = world._get_store("Team")
    flag_store = world._get_store("Flag")
    rew_store = world._get_store("Reward")

    agent_idx = jnp.where(team_store.alive_mask)[0]
    flag_idx = jnp.where(flag_store.alive_mask)[0]
    if (agent_idx.size == 0) or (flag_idx.size == 0):
        return world

    agent_pos = pos_store.read(agent_idx)
    agent_team = team_store.read(agent_idx)

    flag_data = flag_store.read(flag_idx)
    flag_pos = pos_store.read(flag_idx)
    owner = flag_data[:, 0]
    carried = flag_data[:, 1].astype(bool)
    carrier_ids = flag_data[:, 2]

    # Capture flags that are not currently carried
    can_capture = jnp.logical_not(carried)
    if jnp.any(can_capture):
        free_flags = jnp.where(can_capture)[0]
        fpos = flag_pos[free_flags]
        capture_mat = jnp.all(
            agent_pos[:, None, :] == fpos[None, :, :], axis=-1
        )
        enemy = agent_team[:, None] != owner[free_flags][None, :]
        capture_mat = jnp.logical_and(capture_mat, enemy)
        any_capture = jnp.any(capture_mat, axis=0)
        captor_idx = jnp.argmax(capture_mat, axis=0)
        carrier_entity = agent_idx[captor_idx]
        new_carried = jnp.where(any_capture, True, carried[free_flags])
        new_carrier_ids = jnp.where(any_capture, carrier_entity, carrier_ids[free_flags])
        carried = carried.at[free_flags].set(new_carried)
        carrier_ids = carrier_ids.at[free_flags].set(new_carrier_ids)

    # Move carried flags with their carriers
    carried_pos = pos_store.read(carrier_ids)
    flag_pos = jnp.where(carried[:, None], carried_pos, flag_pos)

    # Check for scoring when carrier reaches its base
    base_positions = inputs.get("base_positions")
    if base_positions is None:
        base_positions = jnp.zeros((2, 2), jnp.int32)
    carrier_team = team_store.read(carrier_ids)
    carrier_base = base_positions[carrier_team]
    scored = jnp.logical_and(
        carried,
        jnp.all(carried_pos == carrier_base, axis=1),
    )
    if jnp.any(scored):
        # Award reward to carriers
        rew = rew_store.read(carrier_ids)
        rew = rew + jnp.where(scored, 1.0, 0.0)
        rew_store = rew_store.write(carrier_ids, rew)
        # Reset flags to home positions
        home_pos = base_positions[owner]
        flag_pos = jnp.where(scored[:, None], home_pos, flag_pos)
        carried = jnp.where(scored, False, carried)
        carrier_ids = jnp.where(scored, -1, carrier_ids)

    flag_store = flag_store.write(
        flag_idx,
        jnp.stack(
            [owner, carried.astype(jnp.int32), carrier_ids], axis=1
        ),
    )
    pos_store = pos_store.write(flag_idx, flag_pos)
    world = world._with_store("Flag", flag_store)
    world = world._with_store("Position", pos_store)
    world = world._with_store("Reward", rew_store)
    return world

