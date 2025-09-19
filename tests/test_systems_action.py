import numpy as np
import jax
import jax.numpy as jnp

from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.world import WorldState
from ecsx.systems.action import _entity_indices, _apply_collision, grid_movement_system


def test_systems_package_exports_grid_movement():
    import ecsx.systems as systems_module

    assert systems_module.grid_movement_system is grid_movement_system


def _setup_world() -> WorldState:
    world = WorldState.create(capacity=3, key=jax.random.PRNGKey(0))
    position_spec = ComponentSpecification("Position", jnp.zeros((2,), dtype=jnp.float32))
    action_spec = ComponentSpecification("DiscreteAction", jnp.zeros((), dtype=jnp.int32))
    obstacle_spec = ComponentSpecification("Obstacle", jnp.zeros((), dtype=jnp.int32))
    for spec in (position_spec, action_spec, obstacle_spec):
        world = world.register_component(spec)
    return world


def test_entity_indices_returns_static_range():
    world = WorldState.create(capacity=4, key=jax.random.PRNGKey(1))
    indices = _entity_indices(world)
    np.testing.assert_array_equal(np.asarray(indices), np.arange(4, dtype=np.int32))


def test_apply_collision_reverts_moves_into_obstacles():
    world = _setup_world()
    world = world.add_component_to_entity("Position", 0, jnp.array([0.0, 0.0], dtype=jnp.float32))
    world = world.add_component_to_entity("Position", 1, jnp.array([0.0, 1.0], dtype=jnp.float32))
    world = world.add_component_to_entity("Obstacle", 1, jnp.array(1, dtype=jnp.int32))

    idx = _entity_indices(world)
    pos = world._get_store("Position").read(idx)
    new_pos = pos.at[0].set(jnp.array([0.0, 1.0], dtype=jnp.float32))

    resolved = _apply_collision(world, idx, new_pos, pos)
    np.testing.assert_allclose(np.asarray(resolved[0]), np.asarray(pos[0]))
    np.testing.assert_allclose(np.asarray(resolved[1]), np.asarray(new_pos[1]))


def test_grid_movement_system_updates_positions_and_actions():
    world = _setup_world()
    world = world.add_component_to_entity("Position", 0, jnp.array([0.0, 0.0], dtype=jnp.float32))
    world = world.add_component_to_entity("Position", 1, jnp.array([0.0, 1.0], dtype=jnp.float32))
    world = world.add_component_to_entity("Obstacle", 1, jnp.array(1, dtype=jnp.int32))
    world = world.add_component_to_entity("DiscreteAction", 0, jnp.array(1, dtype=jnp.int32))

    result = grid_movement_system(world, jax.random.PRNGKey(2), inputs={})

    idx = _entity_indices(world)
    positions = result._get_store("Position").read(idx)
    actions = result._get_store("DiscreteAction").read(idx)

    np.testing.assert_allclose(np.asarray(positions[0]), np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(actions[0]), np.array(0, dtype=np.int32))

    np.testing.assert_allclose(np.asarray(positions[1]), np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(actions[1]), np.array(0, dtype=np.int32))
    np.testing.assert_allclose(np.asarray(positions[2]), np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(actions[2]), np.array(0, dtype=np.int32))
