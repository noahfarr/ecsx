from typing import Mapping, Sequence
import jax.numpy as jnp
from ecsx.core.world import World
from ecsx.core.system import SystemFn
from ecsx.core.typing import Array


class Environment:
    """Environment wrapper supporting single or multiple agents."""

    def __init__(
        self,
        world: World,
        systems: Sequence[SystemFn],
        action_component: str,
        agent_indices: Sequence[int] | None = None,
        default_inputs: Mapping[str, Array] | None = None,
    ):
        self.world = world
        self.world.add_systems(*systems)
        self.action_component = action_component
        self.agent_indices = (
            jnp.asarray(agent_indices, jnp.int32)
            if agent_indices is not None
            else jnp.array([], jnp.int32)
        )
        self.default_inputs = default_inputs or {}

    def _indices(self) -> jnp.ndarray:
        if self.agent_indices.size == 0:
            idx = jnp.where(self.world.state.alive_mask)[0].astype(jnp.int32)
            self.agent_indices = idx
        return self.agent_indices

    def reset(self, inputs: Mapping[str, Array] | None = None) -> Array:
        self.world.step(inputs or self.default_inputs)
        idx = self._indices()
        obs_store = self.world.get_store("Observation")
        obs = obs_store.read(idx)
        return obs[0] if idx.size == 1 else obs

    def step(self, action: Array, inputs: Mapping[str, Array] | None = None):
        idx = self._indices()
        act_store = self.world.get_store(self.action_component)
        value = jnp.asarray(action, act_store.spec.dtype).reshape(
            (idx.size, *act_store.spec.shape)
        )
        act_store = act_store.write(idx, value)
        self.world._world = self.world.state._with_store(
            self.action_component, act_store
        )
        self.world.step(inputs or self.default_inputs)
        obs_store = self.world.get_store("Observation")
        rew_store = self.world.get_store("Reward")
        term_store = self.world.get_store("Termination")
        obs = obs_store.read(idx)
        rew = rew_store.read(idx)
        done = term_store.read(idx)
        if idx.size == 1:
            return obs[0], float(rew[0]), bool(done[0]), {}
        return obs, rew, done, {}
