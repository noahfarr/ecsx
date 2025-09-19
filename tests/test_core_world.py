import numpy as np
import pytest
import jax
import jax.numpy as jnp

from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.world import World, WorldState
from ecsx.core.event_specification import EventSpecification


def _array_spec(name: str, shape: tuple[int, ...], dtype=jnp.float32) -> ComponentSpecification:
    return ComponentSpecification(name, jnp.zeros(shape, dtype=dtype))


def test_world_state_component_registration_and_access():
    world = WorldState.create(capacity=3, key=jax.random.PRNGKey(0))
    pos_spec = _array_spec("Position", (2,), jnp.float32)
    world = world.register_component(pos_spec)

    assert "Position" in world.component_stores
    with pytest.raises(ValueError):
        world.register_component(pos_spec)

    with pytest.raises(KeyError):
        world._get_store("Velocity")


def test_world_state_add_remove_component_and_alive_mask():
    world = WorldState.create(capacity=2, key=jax.random.PRNGKey(1))
    pos_spec = _array_spec("Position", (2,), jnp.float32)
    world = world.register_component(pos_spec)

    world = world.add_component_to_entity("Position", 1, jnp.array([1.0, 2.0], dtype=jnp.float32))
    store = world._get_store("Position")
    values = store.read(jnp.array([1], dtype=jnp.int32))
    np.testing.assert_allclose(np.asarray(values), np.array([[1.0, 2.0]], dtype=np.float32))
    assert bool(world.alive_mask[1])

    world = world.remove_component_from_entity("Position", 1)
    values = world._get_store("Position").read(jnp.array([1], dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(values), np.zeros((1, 2), dtype=np.float32))
    assert bool(world.alive_mask[1])


def test_world_state_event_buffers():
    world = WorldState.create(capacity=2, key=jax.random.PRNGKey(2))
    spec = EventSpecification("Log", shape=(1,), dtype=jnp.int32, capacity=2)
    world = world.register_event_buffer(spec)
    with pytest.raises(ValueError):
        world.register_event_buffer(spec)

    payloads = jnp.arange(2, dtype=jnp.int32).reshape(2, 1)
    world = world.write_event_buffer("Log", payloads, jnp.array(2, dtype=jnp.int32))
    assert int(world.event_buffers["Log"].count) == 2

    world = world.reset_event_buffers()
    assert int(world.event_buffers["Log"].count) == 0


def test_world_state_with_alive_mask_validation():
    world = WorldState.create(capacity=2, key=jax.random.PRNGKey(3))
    with pytest.raises(ValueError):
        world.with_alive_mask(jnp.ones((3,), dtype=bool))


def test_world_state_tree_roundtrip():
    world = WorldState.create(capacity=2, key=jax.random.PRNGKey(4))
    pos_spec = _array_spec("Position", (2,), jnp.float32)
    world = world.register_component(pos_spec)
    world = world.add_component_to_entity("Position", 0, jnp.array([3.0, 4.0], dtype=jnp.float32))
    spec = EventSpecification("Log", shape=(1,), dtype=jnp.int32, capacity=2)
    world = world.register_event_buffer(spec)

    children, aux = world.tree_flatten()
    restored = WorldState.tree_unflatten(aux, children)

    np.testing.assert_array_equal(np.asarray(world.alive_mask), np.asarray(restored.alive_mask))
    np.testing.assert_array_equal(
        np.asarray(world._get_store("Position").data),
        np.asarray(restored._get_store("Position").data),
    )
    assert world.capacity == restored.capacity


def test_world_component_and_entity_management():
    world = World(capacity=3, key=jax.random.PRNGKey(5))
    pos_spec = _array_spec("Position", (2,), jnp.float32)
    vel_spec = _array_spec("Velocity", (2,), jnp.float32)
    world.register_component(pos_spec).register_component(vel_spec)

    eid = world.spawn(Position=jnp.array([1.0, 0.0], dtype=jnp.float32))
    assert bool(world.state.alive_mask[eid])
    np.testing.assert_allclose(
        np.asarray(world.get_store("Position").read(jnp.array([eid], dtype=jnp.int32))),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(RuntimeError):
        world.attach_component(1, "Velocity", jnp.array([0.0, 0.0], dtype=jnp.float32))

    world.attach_component(eid, "Velocity", jnp.array([0.5, 0.5], dtype=jnp.float32))
    world.detach_component(eid, "Velocity")
    np.testing.assert_array_equal(
        np.asarray(world.get_store("Velocity").alive_mask),
        np.array([False, False, False]),
    )

    world.despawn(eid)
    assert not bool(world.state.alive_mask[eid])
    np.testing.assert_array_equal(
        np.asarray(world.get_store("Position").alive_mask),
        np.zeros((3,), dtype=bool),
    )


def test_world_add_systems_and_step_updates_state():
    world = World(capacity=1, key=jax.random.PRNGKey(6))
    world.add_systems(lambda state, key, inputs: state)

    before = world.state.time_step
    world.step(inputs={})
    assert world.state.time_step == before + jnp.array(1, dtype=before.dtype)
