import jax
import jax.numpy as jnp

from ecsx.core.world_state import WorldState
from ecsx.core.entity_registry import EntityRegistry
from ecsx.core.component_specification import ComponentSpecification
from ecsx.events.event_specification import EventSpecification
from ecsx.systems.pipeline_builder import build_step
from ecsx.query.query_engine import select


def test_event_buffer_register_write_and_clear():
    key = jax.random.PRNGKey(0)
    world = WorldState.create(capacity=4, key=key)
    spec = EventSpecification("Tick", (1,), jnp.int32, capacity=8)
    world = world.register_event_buffer(spec)

    payloads = jnp.array([[1],[2],[3],[4],[0],[0],[0],[0]], jnp.int32)  # capacity rows
    world = world.write_event_buffer("Tick", payloads, jnp.array(4, jnp.int32))
    assert int(world.event_buffers["Tick"].count) == 4
    world = world.reset_event_buffers()
    assert int(world.event_buffers["Tick"].count) == 0


def test_pipeline_runs_systems_and_updates_components():
    key = jax.random.PRNGKey(0)
    world = WorldState.create(capacity=6, key=key)
    reg = EntityRegistry.create(capacity=6)

    # register a scalar component and an event buffer
    val_spec = ComponentSpecification("Value", (), jnp.int32, jnp.array(0, jnp.int32))
    world = world.register_component(val_spec)
    evt_spec = EventSpecification("Updated", (1,), jnp.int32, capacity=16)
    world = world.register_event_buffer(evt_spec)

    # spawn two entities and attach Value
    e0 = reg.spawn(); e1 = reg.spawn()
    world = world.with_alive_mask(reg.alive_mask)
    world = world.add_component_to_entity("Value", e0, jnp.array(1, jnp.int32))
    world = world.add_component_to_entity("Value", e1, jnp.array(10, jnp.int32))

    # define a simple system: increment all alive Value by 1 and emit their ids
    def increment_and_emit(world, key, inputs):
        mask = select(world, required=("Value",))

        # Make idx length == event buffer capacity
        bufcap = world.event_buffers["Updated"].specification.capacity
        idx = jnp.nonzero(mask, size=bufcap, fill_value=-1)[0]
        count = jnp.minimum(jnp.sum(mask), jnp.array(bufcap, jnp.int32))

        vals = world.component_stores["Value"].read(idx[:count]) + 1
        table = world.component_stores["Value"].write(idx[:count], vals)
        world = world._with_store("Value", table)

        # Now reshape safely
        payloads = idx.reshape((bufcap, 1))
        world = world.write_event_buffer("Updated", payloads, count)
        return world


    step = build_step((increment_and_emit,))

    # run one step
    world = step(world, inputs={})

    got0 = int(world.component_stores["Value"].read(jnp.asarray([e0]))[0])
    got1 = int(world.component_stores["Value"].read(jnp.asarray([e1]))[0])
    assert got0 == 2 and got1 == 11
    assert int(world.event_buffers["Updated"].count) == 2

