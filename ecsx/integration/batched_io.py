# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from ecsx.core.static_world import StaticWorld


def _index_of(sw_batched: StaticWorld, name: str) -> int:
    # Python-side resolution; constant at trace time.
    try:
        return {n: i for i, n in enumerate(sw_batched.store_names)}[name]
    except KeyError:
        raise KeyError(f"Component '{name}' not found in StaticWorld.store_names={sw_batched.store_names}")


def component_indices(sw_batched: StaticWorld, names: Tuple[str, ...]) -> Dict[str, int]:
    """Resolve component names → tuple indices once; pass ints to jitted code."""
    return {n: _index_of(sw_batched, n) for n in names}


def find_agent_indices(
    sw_batched: StaticWorld,
    *,
    observation_name: str = "Observation",
    action_name: str = "Action",
) -> jnp.ndarray:
    """
    Returns: agent_idx [B], the first alive entity that has both Observation and Action.
    If none alive in an env, the index falls back to 0 for that env.
    """
    obs_i = _index_of(sw_batched, observation_name)
    act_i = _index_of(sw_batched, action_name)

    alive = sw_batched.alive_mask         # (B, C)
    has_obs = sw_batched.stores[obs_i].alive_mask  # (B, C)
    has_act = sw_batched.stores[act_i].alive_mask  # (B, C)
    mask = alive & has_obs & has_act                  # (B, C)

    capacity = mask.shape[1]
    scores = jnp.where(mask, jnp.arange(capacity, dtype=jnp.int32), -1)
    # argmax returns 0 if all -1; that's acceptable as a fallback
    idx = jnp.argmax(scores, axis=1)  # (B,)
    return idx


def read_observations_batched_by_index(
    sw_batched: StaticWorld,
    obs_index: int,
    agent_idx: jnp.ndarray,
) -> jnp.ndarray:
    """
    Gather observations for each env's agent.
    sw_batched: StaticWorld with batched leaves (B, ...)
    obs_index: tuple index of Observation store in sw_batched.stores
    agent_idx: [B] entity index per env
    Returns: obs [B, obs_dim]
    """
    obs_store = sw_batched.stores[obs_index]
    data = obs_store.data  # (B, C, D)
    B = data.shape[0]
    gathered = jax.vmap(lambda arr, i: arr[i])(data, agent_idx)  # (B, D)
    return gathered


def write_actions_batched_by_index(
    sw_batched: StaticWorld,
    act_index: int,
    act_name: str,
    agent_idx: jnp.ndarray,
    actions: jnp.ndarray,
) -> StaticWorld:
    """
    Write per-env actions into the 'Action' component at [env, agent_idx, :].
    actions: [B, act_dim]
    Returns updated StaticWorld (batched).
    """
    store = sw_batched.stores[act_index]
    data = store.data  # (B, C, A)

    def _set_one(arr, i, a):
        return arr.at[i].set(a)

    new_data = jax.vmap(_set_one)(data, agent_idx, actions)  # (B, C, A)
    new_store = type(store)(store.spec, new_data, store.alive_mask)
    return sw_batched._with_store(act_name, new_store)


def read_rewards_terms_and_zero_batched_by_index(
    sw_batched: StaticWorld,
    reward_index: int,
    reward_name: str,
    term_index: int,
    agent_idx: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, StaticWorld]:
    """
    Read reward and termination flag per env for agent_idx, and zero the reward in-place (functionally).
    Returns: (reward[B], termination[B], updated StaticWorld)
    """
    # Reward
    r_store = sw_batched.stores[reward_index]  # (B, C, 1)
    r_data = r_store.data

    def _read(arr, i):
        return arr[i, 0]  # scalar

    reward = jax.vmap(_read)(r_data, agent_idx)  # (B,)

    # zero rewards at those indices
    def _zero_one(arr, i):
        return arr.at[i, 0].set(0.0)

    r_new = jax.vmap(_zero_one)(r_data, agent_idx)
    r_store2 = type(r_store)(r_store.spec, r_new, r_store.alive_mask)
    sw2 = sw_batched._with_store(reward_name, r_store2)

    # Termination
    t_store = sw2.stores[term_index]  # (B, C, 1) bool
    t_data = t_store.data

    def _readb(arr, i):
        return arr[i, 0]

    term = jnp.asarray(jax.vmap(_readb)(t_data, agent_idx), jnp.bool_)  # (B,)
    return reward, term, sw2

