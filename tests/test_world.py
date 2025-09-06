import jax
import jax.numpy as jnp
from ecsx.core.world import World
from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.event_specification import EventSpecification


def test_world_register_spawn_attach_detach():
    spec = ComponentSpecification(
        "Position", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32)
    )
    world = World(capacity=4, key=jax.random.PRNGKey(0))
    world.register_component(spec)
    eid = world.spawn()
    world.attach_component(eid, "Position", jnp.array([1.0, 2.0], jnp.float32))
    store = world.get_store("Position")
    assert jnp.array_equal(
        store.read(jnp.array([eid])), jnp.array([[1.0, 2.0]], jnp.float32)
    )
    world.detach_component(eid, "Position")
    store = world.get_store("Position")
    assert jnp.array_equal(
        store.read(jnp.array([eid])), jnp.array([[0.0, 0.0]], jnp.float32)
    )


def test_world_step_resets_event_buffers_and_increments_time():
    world = World(capacity=1, key=jax.random.PRNGKey(0))
    evt_spec = EventSpecification("Evt", (1,), jnp.int32, capacity=1)
    world.register_event_buffer(evt_spec)
    world._world = world.state.write_event_buffer(
        "Evt", jnp.array([[5]], jnp.int32), jnp.array(1, jnp.int32)
    )
    assert int(world.state.event_buffers["Evt"].count) == 1
    world.step()
    assert int(world.state.event_buffers["Evt"].count) == 0
    assert int(world.state.time_step) == 1
