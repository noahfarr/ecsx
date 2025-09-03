import jax
import jax.numpy as jnp

from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.world_state import WorldState
from ecsx.core.entity_registry import EntityRegistry
from ecsx.query.query_engine import select, indices_from_mask
from ecsx.query.selector_language import Has, AllOf


def setup_world(capacity=8):
    key = jax.random.PRNGKey(0)
    world = WorldState.create(capacity=capacity, key=key)
    pos = ComponentSpecification(
        "Position", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32)
    )
    vel = ComponentSpecification(
        "Velocity", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32)
    )
    tag = ComponentSpecification("Tag", (), jnp.bool_, jnp.array(False))
    world = world.register_component(pos)
    world = world.register_component(vel)
    world = world.register_component(tag)
    return world


def test_select_required_and_forbidden_with_alive_mask():
    world = setup_world(capacity=6)
    reg = EntityRegistry.create(capacity=6)

    # Spawn three entities: e0, e1, e2
    e0 = reg.spawn()
    e1 = reg.spawn()
    e2 = reg.spawn()
    world = world.with_alive_mask(reg.alive_mask)

    # Attach components:
    world = world.add_component_to_entity("Position", e0, jnp.array([1.0, 2.0]))
    world = world.add_component_to_entity("Velocity", e0, jnp.array([0.1, 0.2]))
    world = world.add_component_to_entity("Position", e1, jnp.array([3.0, 4.0]))
    world = world.add_component_to_entity("Tag", e1, jnp.array(True))
    world = world.add_component_to_entity("Position", e2, jnp.array([5.0, 6.0]))
    world = world.add_component_to_entity("Velocity", e2, jnp.array([0.0, 0.0]))
    world = world.add_component_to_entity("Tag", e2, jnp.array(True))

    # Query: alive & Has(Position) & Has(Velocity) & Without(Tag)
    mask = select(world, required=("Position", "Velocity"), forbidden=("Tag",))
    idx = indices_from_mask(mask)
    assert idx.shape[0] == 1 and int(idx[0]) == e0

    # Despawn e0 -> should remove it from selection
    reg.despawn(e0)
    world = world.with_alive_mask(reg.alive_mask)
    mask2 = select(world, required=("Position", "Velocity"), forbidden=("Tag",))
    assert jnp.sum(mask2) == 0


def test_selector_language_compile_and_run():
    world = setup_world(capacity=5)
    reg = EntityRegistry.create(capacity=5)
    e0 = reg.spawn()
    e1 = reg.spawn()
    world = world.with_alive_mask(reg.alive_mask)
    world = world.add_component_to_entity("Position", e0, jnp.array([1.0, 0.0]))
    world = world.add_component_to_entity("Velocity", e0, jnp.array([0.5, 0.0]))
    world = world.add_component_to_entity("Position", e1, jnp.array([2.0, 0.0]))
    world = world.add_component_to_entity("Tag", e1, jnp.array(True))

    sel = AllOf("Position", "Velocity") - Has("Tag")
    fn = sel.compile()
    mask = fn(world)
    idx = indices_from_mask(mask)
    assert idx.shape[0] == 1 and int(idx[0]) == e0
