import jax
import jax.numpy as jnp

from ecsx.environments import build_capture_the_flag


def test_capture_the_flag_capture_and_score():
    env = build_capture_the_flag(team_size=1, map_size=(3, 1))
    state, _ = env.reset(jax.random.PRNGKey(0))
    # Move agent 0 to enemy flag (right side)
    actions = jnp.array([4, 0], jnp.int32)
    state, _ = env.step(jax.random.PRNGKey(1), state, actions)
    state, _ = env.step(jax.random.PRNGKey(2), state, actions)
    # Return to base with the flag
    back = jnp.array([3, 0], jnp.int32)
    state, _ = env.step(jax.random.PRNGKey(3), state, back)
    state, ts = env.step(jax.random.PRNGKey(4), state, back)
    assert ts.reward[0] == 1.0
    # Flag should be reset at home position and not carried
    flag_store = state.world._get_store("Flag")
    flag_data = flag_store.read(jnp.where(flag_store.alive_mask)[0])
    assert jnp.all(flag_data[:, 1] == 0)
