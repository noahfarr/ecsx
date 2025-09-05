# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from ..world.world_api import World
from ..systems.pipeline_builder import build_step_unrolled
from ..core.typing import Array


SystemFn = Callable[..., World]


class VectorWorldRunner:
    """
    Host-side runner for N independent Worlds.
    - Builds one jitted, unrolled step function and reuses it for all envs.
    - Assumes single 'agent' per world: first alive entity with Observation + Action.
    - Works with the same systems you already use (Action->Velocity->Physics->Observation).
    """

    def __init__(
        self,
        world_builder: Callable[[], World],
        step_systems: Tuple[SystemFn, ...],
        num_envs: int,
        unroll: int = 8,
    ):
        assert num_envs >= 1
        self.envs: List[World] = [world_builder() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.unroll = int(unroll)
        self._step_fn = build_step_unrolled(step_systems, unroll=self.unroll)
        self._agent_ids: List[Optional[int]] = [None] * num_envs

        # infer dims from env 0
        obs_dim = int(self.envs[0].state.component_stores["Observation"].spec.shape[0])
        act_dim = int(self.envs[0].state.component_stores["Action"].spec.shape[0])
        self.observation_dimension = obs_dim
        self.action_dimension = act_dim

    # ---------- agent selection per env ----------
    def _get_agent_id(self, env_idx: int) -> int:
        if self._agent_ids[env_idx] is not None:
            return int(self._agent_ids[env_idx])
        w = self.envs[env_idx].state
        alive = w.alive_mask
        has_obs = w.component_stores["Observation"].alive_mask
        has_act = w.component_stores["Action"].alive_mask
        mask = jnp.logical_and(alive, jnp.logical_and(has_obs, has_act))
        idx = jnp.nonzero(mask)[0]
        if idx.size == 0:
            raise RuntimeError(f"[env {env_idx}] No entity with Observation + Action.")
        self._agent_ids[env_idx] = int(idx[0])
        return self._agent_ids[env_idx]  # type: ignore[return-value]

    # ---------- I/O ----------
    def write_actions(self, actions: np.ndarray | jnp.ndarray) -> None:
        """
        actions: (num_envs, action_dim) in [-1,1]
        """
        if isinstance(actions, jnp.ndarray):
            actions = np.asarray(actions, np.float32)
        assert actions.shape == (self.num_envs, self.action_dimension)
        for i, env in enumerate(self.envs):
            agent = self._get_agent_id(i)
            act_store = env.state.component_stores["Action"]
            act_store = act_store.write(jnp.asarray([agent]), jnp.asarray([actions[i]], jnp.float32))
            env._world = env.state._with_store("Action", act_store)

    def read_observations(self) -> np.ndarray:
        """
        Returns (num_envs, obs_dim)
        """
        bufs = []
        for i, env in enumerate(self.envs):
            agent = self._get_agent_id(i)
            obs = env.state.component_stores["Observation"].read(jnp.asarray([agent]))[0]
            bufs.append(np.asarray(obs, np.float32))
        return np.stack(bufs, axis=0)

    def read_rewards_and_clear(self) -> np.ndarray:
        """
        Returns (num_envs,) and clears reward for next step.
        """
        out = np.zeros((self.num_envs,), np.float32)
        for i, env in enumerate(self.envs):
            agent = self._get_agent_id(i)
            rew_store = env.state.component_stores["Reward"]
            r = float(rew_store.read(jnp.asarray([agent]))[0, 0])
            out[i] = r
            zero = jnp.zeros_like(rew_store.read(jnp.asarray([agent])))
            rew_store = rew_store.write(jnp.asarray([agent]), zero)
            env._world = env.state._with_store("Reward", rew_store)
        return out

    def read_terminations(self) -> np.ndarray:
        out = np.zeros((self.num_envs,), np.bool_)
        for i, env in enumerate(self.envs):
            agent = self._get_agent_id(i)
            t = bool(env.state.component_stores["Termination"].read(jnp.asarray([agent]))[0, 0])
            out[i] = t
        return out

    # ---------- stepping ----------
    def step(self, inputs: Mapping[str, Array] | None = None) -> None:
        """
        Advance each world by `unroll` ECS steps via the shared jitted function.
        """
        if inputs is None:
            inputs = {}
        for i in range(self.num_envs):
            env = self.envs[i]
            env._world = self._step_fn(env.state, inputs)

    def block_until_ready(self) -> None:
        # Sync on a cheap leaf; any env is fine.
        jax.block_until_ready(self.envs[-1].state.time_step)

