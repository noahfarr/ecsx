import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from ecsx.environments import build_grid_world
from ecsx.algorithms import PPOAgent, PPOConfig


def test_ppo_train_step_updates_params():
    env = build_grid_world(num_agents=1, grid_size=(5, 5), goal_position=(4, 4))
    config = PPOConfig(batch_size=8, epochs=1)
    agent = PPOAgent(obs_dim=2, act_dim=5, config=config, key=jax.random.PRNGKey(0))
    flat_before, _ = ravel_pytree(agent.params)
    key = jax.random.PRNGKey(1)
    agent.train_step(env, key)
    flat_after, _ = ravel_pytree(agent.params)
    assert not jnp.allclose(flat_before, flat_after)


def test_ppo_train_step_updates_params_multi_agent():
    env = build_grid_world(num_agents=2, grid_size=(5, 5), goal_position=(4, 4))
    config = PPOConfig(batch_size=8, epochs=1)
    agent = PPOAgent(obs_dim=2, act_dim=5, config=config, key=jax.random.PRNGKey(0))
    flat_before, _ = ravel_pytree(agent.params)
    key = jax.random.PRNGKey(1)
    agent.train_step(env, key)
    flat_after, _ = ravel_pytree(agent.params)
    assert not jnp.allclose(flat_before, flat_after)
