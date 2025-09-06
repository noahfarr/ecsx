from typing import Mapping, Sequence
import jax.numpy as jnp
from ecsx.core.world import World
from ecsx.core.system import SystemFn
from ecsx.core.typing import Array


class Environment:
    """Minimal environment wrapper around a World."""

    def __init__(
        self, world: World, systems: Sequence[SystemFn], action_component: str
    ):
        self.world = world
        self.world.add_systems(*systems)
        self.action_component = action_component

    def reset(self, inputs: Mapping[str, Array] | None = None) -> Array:
        self.world.step(inputs)
        obs_store = self.world.get_store("Observation")
        return obs_store.read(jnp.array([0], jnp.int32))[0]

    def step(self, action: Array, inputs: Mapping[str, Array] | None = None):
        act_store = self.world.get_store(self.action_component)
        idx = jnp.array([0], jnp.int32)
        value = jnp.asarray(action, act_store.spec.dtype).reshape(
            (1, *act_store.spec.shape)
        )
        act_store = act_store.write(idx, value)
        self.world._world = self.world.state._with_store(
            self.action_component, act_store
        )
        self.world.step(inputs)
        obs = self.world.get_store("Observation").read(idx)[0]
        rew = self.world.get_store("Reward").read(idx)[0]
        done = self.world.get_store("Termination").read(idx)[0]
        return obs, float(rew), bool(done), {}
