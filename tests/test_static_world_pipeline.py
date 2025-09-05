import jax
import jax.numpy as jnp

from ecsx.core.world_state import WorldState
from ecsx.core.entity_registry import EntityRegistry
from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.static_world import StaticWorld
from ecsx.systems.pipeline_builder import build_step_static
from ecsx.systems.action_application_system import action_application_system
from ecsx.systems.physics_2d import physics_2d
from ecsx.systems.observation_system import observation_system

def test_static_pipeline_updates_components():
    key = jax.random.PRNGKey(0)
    w = WorldState.create(capacity=8, key=key)
    reg = EntityRegistry.create(capacity=8)

    pos = ComponentSpecification("Position", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32))
    vel = ComponentSpecification("Velocity", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32))
    act = ComponentSpecification("Action", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32))
    obs = ComponentSpecification("Observation", (4,), jnp.float32, jnp.zeros((4,), jnp.float32))
    w = w.register_component(pos).register_component(vel).register_component(act).register_component(obs)

    e0 = reg.spawn()
    w = w.with_alive_mask(reg.alive_mask)
    w = w.add_component_to_entity("Position", e0, jnp.array([0.0, 0.0], jnp.float32))
    w = w.add_component_to_entity("Velocity", e0, jnp.array([0.0, 0.0], jnp.float32))
    w = w.add_component_to_entity("Action",   e0, jnp.array([0.3, -0.4], jnp.float32))
    w = w.add_component_to_entity("Observation", e0, jnp.zeros((4,), jnp.float32))

    order = ("Position", "Velocity", "Action", "Observation")
    sw = StaticWorld.freeze(w, order)

    step = build_step_static((action_application_system(velocity_scale=2.0),
                              physics_2d(0.5),
                              observation_system(True)))
    sw = step(sw, {})

    pos_after = sw.stores[order.index("Position")].read(jnp.asarray([e0]))[0]
    vel_after = sw.stores[order.index("Velocity")].read(jnp.asarray([e0]))[0]
    obs_after = sw.stores[order.index("Observation")].read(jnp.asarray([e0]))[0]
    assert jnp.allclose(vel_after, jnp.array([0.6, -0.8], jnp.float32))
    assert jnp.allclose(pos_after, jnp.array([0.3, -0.4], jnp.float32))  # dt=0.5 * vel
    assert obs_after.shape == (4,)

