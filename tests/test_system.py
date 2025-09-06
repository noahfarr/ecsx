import jax
import jax.numpy as jnp
from ecsx.core.world import WorldState
from ecsx.core.system import build_step, set_random_key


def test_set_random_key():
    world = WorldState.create(capacity=1, key=jax.random.PRNGKey(0))
    new_key = jax.random.PRNGKey(1)
    updated = set_random_key(world, new_key)
    assert jnp.array_equal(updated.random_key, new_key)
    assert updated is not world


def test_build_step_increments_time():
    world = WorldState.create(capacity=1, key=jax.random.PRNGKey(0))

    def identity_system(world, key, inputs):
        return world

    step = build_step((identity_system,))
    new_world = step(world, {})
    assert int(new_world.time_step) == 1
