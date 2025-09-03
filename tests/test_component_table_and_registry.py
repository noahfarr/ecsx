import jax
import jax.numpy as jnp

from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.component_table import ComponentTable
from ecsx.core.entity_registry import EntityRegistry
from ecsx.core.world_state import WorldState


def test_component_table_write_and_clear_roundtrip():
    spec = ComponentSpecification(
        "Position", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32)
    )
    table = ComponentTable.from_spec(spec, capacity=4)
    indices = jnp.asarray([1, 3])
    values = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], jnp.float32)
    table2 = table.write(indices, values)
    assert bool(table2.alive_mask[1]) and bool(table2.alive_mask[3])
    assert jnp.allclose(table2.read(jnp.asarray([1]))[0], jnp.array([1.0, 2.0]))
    table3 = table2.clear(indices)
    assert not bool(table3.alive_mask[1]) and not bool(table3.alive_mask[3])


def test_entity_registry_spawn_and_despawn():
    reg = EntityRegistry.create(capacity=3)
    e0 = reg.spawn()
    e1 = reg.spawn()
    assert bool(reg.alive_mask[e0]) and bool(reg.alive_mask[e1])
    reg.despawn(e0)
    assert not bool(reg.alive_mask[e0])
    e2 = reg.spawn()
    # freed id should be reusable
    assert e2 == e0


def test_world_register_and_attach_component():
    key = jax.random.PRNGKey(0)
    world = WorldState.create(capacity=4, key=key)
    pos_spec = ComponentSpecification(
        "Position", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32)
    )
    world = world.register_component(pos_spec)
    world = world.add_component_to_entity(
        "Position", 2, jnp.array([5.0, 6.0], jnp.float32)
    )
    got = world.component_tables["Position"].read(jnp.asarray([2]))[0]
    assert jnp.allclose(got, jnp.array([5.0, 6.0]))
