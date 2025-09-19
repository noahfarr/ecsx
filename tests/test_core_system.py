import numpy as np
import jax
import jax.numpy as jnp

from ecsx.core.event_specification import EventSpecification
from ecsx.core.system import build_step, set_random_key
from ecsx.core.world import WorldState


def test_set_random_key_replaces_state_key():
    world = WorldState.create(capacity=1, key=jax.random.PRNGKey(0))
    new_key = jax.random.PRNGKey(123)
    updated = set_random_key(world, new_key)
    np.testing.assert_array_equal(np.asarray(updated.random_key), np.asarray(new_key))
    np.testing.assert_array_equal(np.asarray(world.random_key), np.asarray(jax.random.PRNGKey(0)))


def test_build_step_runs_systems_with_split_keys_and_resets_buffers():
    world = WorldState.create(capacity=1, key=jax.random.PRNGKey(0))
    spec = EventSpecification(name="Events", shape=(1,), dtype=jnp.int32, capacity=1)
    world = world.register_event_buffer(spec)
    world = world.write_event_buffer(
        "Events", jnp.ones((1, 1), dtype=jnp.int32), jnp.array(1, dtype=jnp.int32)
    )

    calls: list[tuple[str, jax.Array, dict[str, jax.Array]]] = []

    def system_a(world_state, key, inputs):
        calls.append(("a", key, inputs))
        assert int(world_state.event_buffers["Events"].count) == 0
        return world_state.write_event_buffer(
            "Events", jnp.zeros((1, 1), dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)
        )

    def system_b(world_state, key, inputs):
        calls.append(("b", key, inputs))
        return world_state

    step = build_step((system_a, system_b))
    inputs = {"value": jnp.array(42, dtype=jnp.int32)}
    result = step(world, inputs)

    assert result.time_step == jnp.array(1, dtype=jnp.int32)
    assert len(calls) == 2
    assert calls[0][0] == "a"
    assert calls[1][0] == "b"
    assert calls[0][2] is inputs
    assert calls[1][2] is inputs

    key = world.random_key
    key, expected_a = jax.random.split(key)
    key, expected_b = jax.random.split(key)

    np.testing.assert_array_equal(np.asarray(calls[0][1]), np.asarray(expected_a))
    np.testing.assert_array_equal(np.asarray(calls[1][1]), np.asarray(expected_b))
    np.testing.assert_array_equal(np.asarray(result.random_key), np.asarray(key))
    assert int(result.event_buffers["Events"].count) == 0
