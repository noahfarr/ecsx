import jax
import jax.numpy as jnp

from ecsx.world.world_api import World
from ecsx.events.event_specification import EventSpecification

from ecsx.components.position import get_position_specification
from ecsx.components.velocity import get_velocity_specification
from ecsx.components.collider import get_collider_specification
from ecsx.components.reward_on_touch import get_reward_on_touch_specification
from ecsx.components.reward import get_reward_specification
from ecsx.components.termination import get_termination_specification
from ecsx.components.observation import get_observation_specification
from ecsx.components.action import get_action_specification

from ecsx.systems.pipeline_builder import build_step
from ecsx.systems.physics_2d import physics_2d
from ecsx.systems.collision_2d import collision_2d
from ecsx.systems.reward_system import reward_system
from ecsx.systems.termination_system import termination_system
from ecsx.systems.observation_system import observation_system
from ecsx.systems.action_application_system import action_application_system


def test_collision_triggers_reward_increment():
    w = World(capacity=8, key=jax.random.PRNGKey(0))
    w.register_component(get_position_specification())
    w.register_component(get_velocity_specification())
    w.register_component(get_collider_specification())
    w.register_component(get_reward_on_touch_specification())
    w.register_component(get_reward_specification())
    w.register_event_buffer(EventSpecification("Collision", (2,), jnp.int32, capacity=32))

    # Spawn goal (RewardOnTouch) at x=0.1, agent moving toward it from origin
    goal = w.spawn(Position=jnp.array([0.1, 0.0], jnp.float32),
                   Collider=jnp.array([0.1], jnp.float32),
                   RewardOnTouch=jnp.array([5.0], jnp.float32))
    agent = w.spawn(Position=jnp.array([0.0, 0.0], jnp.float32),
                    Velocity=jnp.array([0.2, 0.0], jnp.float32),
                    Collider=jnp.array([0.1], jnp.float32),
                    Reward=jnp.array([0.0], jnp.float32))

    step = build_step((physics_2d(0.5),  # move agent by 0.1
                       collision_2d("Collision"),
                       reward_system("Collision")))
    w = w  # alias
    w._world = step(w.state, {})

    got = float(w.state.component_stores["Reward"].read(jnp.asarray([agent]))[0, 0])
    assert got == 5.0


def test_observation_and_action_update_velocity():
    w = World(capacity=8, key=jax.random.PRNGKey(1))
    w.register_component(get_position_specification())
    w.register_component(get_velocity_specification())
    w.register_component(get_observation_specification(4))
    w.register_component(get_action_specification(2))

    agent = w.spawn(Position=jnp.array([0.0, 0.0], jnp.float32),
                    Velocity=jnp.array([0.0, 0.0], jnp.float32),
                    Observation=jnp.zeros((4,), jnp.float32),
                    Action=jnp.zeros((2,), jnp.float32))

    # Preload actions directly into the component store (simulating a policy output)
    act_store = w.state.component_stores["Action"]
    act_store = act_store.write(jnp.asarray([agent]), jnp.asarray([[0.3, -0.4]], jnp.float32))
    w._world = w.state._with_store("Action", act_store)

    step = build_step((observation_system(True), action_application_system(velocity_scale=2.0)))
    w._world = step(w.state, {})

    vel = w.state.component_stores["Velocity"].read(jnp.asarray([agent]))[0]
    obs = w.state.component_stores["Observation"].read(jnp.asarray([agent]))[0]
    assert jnp.allclose(vel, jnp.array([0.6, -0.8], jnp.float32))
    assert obs.shape == (4,)

