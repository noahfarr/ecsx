from typing import Mapping, Sequence, Tuple

import jax.numpy as jnp
from flax.struct import dataclass

from ecsx.core.system import SystemFn, set_random_key
from ecsx.core.typing import Array, Key
from ecsx.core.world import World, WorldState


@dataclass
class EnvState:
    """Container holding the mutable world state."""

    world: WorldState


@dataclass
class TimeStep:
    """Dataclass returned from ``step`` containing transition data."""

    observation: Array
    reward: Array
    done: Array
    info: dict


class Environment:
    """Functional environment exposing pure ``reset`` and ``step``."""

    def __init__(
        self,
        world: World,
        systems: Sequence[SystemFn],
        action_component: str,
        agent_indices: Sequence[int] | None = None,
        default_inputs: Mapping[str, Array] | None = None,
    ) -> None:
        self.world = world
        self.world.add_systems(*systems)
        self.step_fn = self.world.build_step()
        self.action_component = action_component
        self.agent_indices = (
            jnp.asarray(agent_indices, jnp.int32)
            if agent_indices is not None
            else jnp.array([], jnp.int32)
        )
        self.default_inputs = default_inputs or {}

    def _indices(self, state: EnvState) -> jnp.ndarray:
        if self.agent_indices.size == 0:
            return jnp.where(state.world.alive_mask)[0].astype(jnp.int32)
        return self.agent_indices

    def reset(self, key: Key) -> Tuple[EnvState, TimeStep]:
        """Reset the world using ``key`` and return initial state and observation."""

        world = set_random_key(self.world.state, key)
        world = self.step_fn(world, self.default_inputs)
        state = EnvState(world)
        idx = self._indices(state)
        obs_store = world._get_store("Observation")
        rew_store = world._get_store("Reward")
        term_store = world._get_store("Termination")
        obs = obs_store.read(idx)
        rew = rew_store.read(idx)
        done = term_store.read(idx)
        obs = obs[0] if idx.size == 1 else obs
        rew = rew[0] if idx.size == 1 else rew
        done = done[0] if idx.size == 1 else done
        ts = TimeStep(obs, rew, done, {})
        return state, ts

    def step(
        self, key: Key, state: EnvState, action: Array
    ) -> Tuple[EnvState, TimeStep]:
        """Advance the environment by one step."""

        idx = self._indices(state)
        act_store = state.world._get_store(self.action_component)
        value = jnp.asarray(action, act_store.spec.dtype).reshape(
            (idx.size, *act_store.spec.shape)
        )
        act_store = act_store.write(idx, value)
        world = state.world._with_store(self.action_component, act_store)
        world = set_random_key(world, key)
        world = self.step_fn(world, self.default_inputs)
        state = EnvState(world)
        obs_store = world._get_store("Observation")
        rew_store = world._get_store("Reward")
        term_store = world._get_store("Termination")
        obs = obs_store.read(idx)
        rew = rew_store.read(idx)
        done = term_store.read(idx)
        obs = obs[0] if idx.size == 1 else obs
        rew = rew[0] if idx.size == 1 else rew
        done = done[0] if idx.size == 1 else done
        ts = TimeStep(obs, rew, done, {})
        return state, ts
