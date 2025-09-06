import jax.numpy as jnp
from ecsx.envs import build_grid_world


def test_multi_agent_environment():
    env = build_grid_world(num_agents=2, grid_size=(5, 5))
    obs = env.reset()
    assert obs.shape == (2, 2)
    actions = jnp.array([4, 1], jnp.int32)
    obs, rew, done, _ = env.step(actions)
    assert jnp.array_equal(obs, jnp.array([[1.0, 0.0], [0.0, 2.0]], jnp.float32))
    assert rew.shape == (2,)
    assert done.shape == (2,)
