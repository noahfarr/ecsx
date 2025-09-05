from dataclasses import replace
from typing import Any, Callable, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
import jax.numpy as jnp

from ..world.world_api import World
from ..systems.pipeline_builder import build_step


SystemFn = Callable[..., Any]


class EcsxGymnasiumEnvironment(gym.Env):
    """
    Minimal single-agent Box action/observation Gym wrapper.
    Assumes components exist: Observation, Action, Reward, Termination.
    You provide the world builder and the systems composing one 'step'.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        world_builder: Callable[[], World],
        step_systems: Tuple[SystemFn, ...],
        agent_entity_id: Optional[int] = None,
        observation_dimension: Optional[int] = None,
        action_dimension: Optional[int] = None,
    ):
        self.world: World = world_builder()
        self._step_fn = build_step(step_systems)
        self._agent: Optional[int] = agent_entity_id

        # Infer dims if not provided
        if observation_dimension is None:
            observation_dimension = int(self.world.state.component_stores["Observation"].spec.shape[0])
        if action_dimension is None:
            action_dimension = int(self.world.state.component_stores["Action"].spec.shape[0])

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dimension,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dimension,), dtype=np.float32
        )

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # No world rebuild here; assume caller’s world_builder already sets initial state.
        obs = self._read_observation()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        # 1) write action for the agent
        agent = self._get_agent_id()
        act_store = self.world.state.component_stores["Action"]
        act_store = act_store.write(jnp.asarray([agent]), jnp.asarray([action], jnp.float32))
        self.world._world = self.world.state._with_store("Action", act_store)

        # 2) run one ecsx step
        self.world._world = self._step_fn(self.world.state, {})

        # 3) read obs / reward / termination and clear reward
        obs = self._read_observation()
        rew = float(self.world.state.component_stores["Reward"].read(jnp.asarray([agent]))[0, 0])

        reward_store = self.world.state.component_stores["Reward"]
        zero = jnp.zeros_like(reward_store.read(jnp.asarray([agent])))
        reward_store = reward_store.write(jnp.asarray([agent]), zero)
        self.world._world = self.world.state._with_store("Reward", reward_store)

        term = bool(self.world.state.component_stores["Termination"].read(jnp.asarray([agent]))[0, 0])
        truncated = False
        info = {}
        return obs, rew, term, truncated, info

    # ---------- helpers ----------
    def _get_agent_id(self) -> int:
        if self._agent is not None:
            return int(self._agent)
        # default to first alive entity that has Observation + Action
        alive = self.world.state.alive_mask
        has_obs = self.world.state.component_stores["Observation"].alive_mask
        has_act = self.world.state.component_stores["Action"].alive_mask
        mask = jnp.logical_and(alive, jnp.logical_and(has_obs, has_act))
        idx = jnp.nonzero(mask)[0]
        if idx.size == 0:
            raise RuntimeError("No agent entity with Observation and Action found.")
        self._agent = int(idx[0])
        return self._agent

    def _read_observation(self) -> np.ndarray:
        agent = self._get_agent_id()
        obs = self.world.state.component_stores["Observation"].read(jnp.asarray([agent]))[0]
        return np.asarray(obs, dtype=np.float32)

