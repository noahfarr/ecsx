from typing import Mapping
import jax
import jax.numpy as jnp
from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState


def _alive_indices(world: WorldState) -> jnp.ndarray:
    return jnp.where(world.alive_mask)[0].astype(jnp.int32)


def distance_reward_system(
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


def goal_reward_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _alive_indices(world)
    if idx.size == 0:
        return world
    pos_store = world._get_store("Position")
    rew_store = world._get_store("Reward")
    pos = pos_store.read(idx)
    goal = jnp.asarray(inputs.get("goal_position"), jnp.float32)
    goal = goal.reshape((1, -1))
    reached = jnp.all(pos == goal, axis=1)
    reward = jnp.where(
        reached, jnp.array(1.0, jnp.float32), jnp.array(0.0, jnp.float32)
    )
    rew_store = rew_store.write(idx, reward)
    return world._with_store("Reward", rew_store)


def step_penalty_reward_system(
    world: WorldState, key: Key, inputs: Mapping[str, Array]
) -> WorldState:
    idx = _alive_indices(world)
    if idx.size == 0:
        return world
    rew_store = world._get_store("Reward")
    penalty = float(inputs.get("penalty", 1.0))
    reward = -jnp.full((idx.size,), penalty, jnp.float32)
    rew_store = rew_store.write(idx, reward)
    return world._with_store("Reward", rew_store)
