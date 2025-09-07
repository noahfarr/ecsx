import jax
import jax.numpy as jnp
from ecsx.environments import build_grid_world


def test_multi_agent_environment():
    env = build_grid_world(num_agents=2, grid_size=(5, 5))
    state, ts = env.reset(jax.random.PRNGKey(0))
    assert ts.observation.shape == (2, 2)
    actions = jnp.array([4, 1], jnp.int32)
    state, ts = env.step(jax.random.PRNGKey(1), state, actions)
    assert jnp.array_equal(ts.observation, jnp.array([[1.0, 0.0], [0.0, 2.0]], jnp.float32))
    assert ts.reward.shape == (2,)
    assert ts.done.shape == (2,)
