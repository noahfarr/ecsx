# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Callable, Mapping, Optional, Tuple
import gymnasium as gym
import numpy as np
import jax.numpy as jnp

from ..world.world_api import World
from ..systems.pipeline_builder import build_step
from ..core.typing import Array

SystemFn = Callable[..., World]

class EcsxGymnasiumDiscreteEnvironment(gym.Env):
    """
    Minimal single-agent Discrete(n) action Gym wrapper.
    Assumes components: Observation, DiscreteAction, Reward, Termination.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        world_builder: Callable[[], World],
        step_systems: Tuple[SystemFn, ...],
        agent_entity_id: Optional[int] = None,
        observation_dimension: Optional[int] = None,
        num_actions: int = 5,
    ):
        self.world: World = world_builder()
        self._step_fn = build_step(step_systems)
        self._agent: Optional[int] = agent_entity_id

        if observation_dimension is None:
            observation_dimension = int(self.world.state.component_stores["Observation"].spec.shape[0])

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dimension,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(num_actions)
        self._num_actions = num_actions

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs = self._read_observation()
        return obs, {}

    def step(self, action: int):
        agent = self._get_agent_id()
        act_store = self.world.state.component_stores["DiscreteAction"]
        act_store = act_store.write(jnp.asarray([agent]), jnp.asarray([[int(action)]], jnp.int32))
        self.world._world = self.world.state._with_store("DiscreteAction", act_store)

        self.world._world = self._step_fn(self.world.state, {})

        obs = self._read_observation()
        rew = float(self.world.state.component_stores["Reward"].read(jnp.asarray([agent]))[0, 0])

        reward_store = self.world.state.component_stores["Reward"]
        zero = jnp.zeros_like(reward_store.read(jnp.asarray([agent])))
        reward_store = reward_store.write(jnp.asarray([agent]), zero)
        self.world._world = self.world.state._with_store("Reward", reward_store)

        term = bool(self.world.state.component_stores["Termination"].read(jnp.asarray([agent]))[0, 0])
        truncated = False
        return obs, rew, term, truncated, {}

    def _get_agent_id(self) -> int:
        if self._agent is not None:
            return int(self._agent)
        alive = self.world.state.alive_mask
        has_obs = self.world.state.component_stores["Observation"].alive_mask
        has_act = self.world.state.component_stores["DiscreteAction"].alive_mask
        mask = jnp.logical_and(alive, jnp.logical_and(has_obs, has_act))
        idx = jnp.nonzero(mask)[0]
        if idx.size == 0:
            raise RuntimeError("No agent with Observation + DiscreteAction found.")
        self._agent = int(idx[0])
        return self._agent

    def _read_observation(self) -> np.ndarray:
        agent = self._get_agent_id()
        obs = self.world.state.component_stores["Observation"].read(jnp.asarray([agent]))[0]
        return np.asarray(obs, dtype=np.float32)

