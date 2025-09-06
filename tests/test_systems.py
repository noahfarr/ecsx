import jax
import jax.numpy as jnp
from ecsx.core.world import World
from ecsx.systems import (
    observation_system,
    discrete_action_system,
    continuous_action_system,
    distance_reward_system,
    goal_reward_system,
    step_penalty_reward_system,
    termination_system,
)
from ecsx.components import (
    get_position_specification,
    get_observation_specification,
    get_discrete_action_specification,
    get_continuous_action_specification,
    get_reward_specification,
    get_termination_specification,
    get_obstacle_specification,
)


def make_world(action_spec_fn):
    world = World(capacity=2, key=jax.random.PRNGKey(0))
    world.register_component(get_position_specification())
    world.register_component(get_observation_specification())
    world.register_component(action_spec_fn())
    world.register_component(get_reward_specification())
    world.register_component(get_termination_specification())
    world.register_component(get_obstacle_specification())
    return world


def test_observation_system():
    world = make_world(get_discrete_action_specification)
    eid = world.spawn(Position=jnp.array([1.0, 2.0], jnp.float32))
    world.add_systems(observation_system).step()
    obs = world.get_store("Observation").read(jnp.array([eid]))
    assert jnp.array_equal(obs, jnp.array([[1.0, 2.0]], jnp.float32))


def test_discrete_action_system():
    world = make_world(get_discrete_action_specification)
    eid = world.spawn(
        Position=jnp.array([0.0, 0.0], jnp.float32),
        DiscreteAction=jnp.array(4, jnp.int32),
    )
    world.add_systems(discrete_action_system).step()
    pos = world.get_store("Position").read(jnp.array([eid]))
    act = world.get_store("DiscreteAction").read(jnp.array([eid]))
    assert jnp.array_equal(pos, jnp.array([[1.0, 0.0]], jnp.float32))
    assert jnp.array_equal(act, jnp.array([0], jnp.int32))


def test_continuous_action_system():
    world = make_world(get_continuous_action_specification)
    eid = world.spawn(
        Position=jnp.array([0.0, 0.0], jnp.float32),
        ContinuousAction=jnp.array([1.0, 2.0], jnp.float32),
    )
    world.add_systems(continuous_action_system).step()
    pos = world.get_store("Position").read(jnp.array([eid]))
    act = world.get_store("ContinuousAction").read(jnp.array([eid]))
    assert jnp.array_equal(pos, jnp.array([[1.0, 2.0]], jnp.float32))
    assert jnp.array_equal(act, jnp.array([[0.0, 0.0]], jnp.float32))


def test_collision_handling():
    world = make_world(get_discrete_action_specification)
    world.spawn(Position=jnp.array([1.0, 0.0], jnp.float32), Obstacle=jnp.array(True))
    eid = world.spawn(
        Position=jnp.array([0.0, 0.0], jnp.float32),
        DiscreteAction=jnp.array(4, jnp.int32),
    )
    world.add_systems(discrete_action_system).step()
    pos = world.get_store("Position").read(jnp.array([eid]))
    assert jnp.array_equal(pos, jnp.array([[0.0, 0.0]], jnp.float32))


def test_distance_reward_system():
    world = make_world(get_discrete_action_specification)
    eid = world.spawn(Position=jnp.array([3.0, 4.0], jnp.float32))
    world.add_systems(distance_reward_system).step()
    rew = world.get_store("Reward").read(jnp.array([eid]))
    assert jnp.allclose(rew, jnp.array([-5.0], jnp.float32))


def test_goal_reward_system():
    world = make_world(get_discrete_action_specification)
    eid = world.spawn(Position=jnp.array([1.0, 1.0], jnp.float32))
    world.add_systems(goal_reward_system).step({"goal_position": jnp.array([1.0, 1.0], jnp.float32)})
    rew = world.get_store("Reward").read(jnp.array([eid]))
    assert jnp.array_equal(rew, jnp.array([1.0], jnp.float32))


def test_step_penalty_reward_system():
    world = make_world(get_discrete_action_specification)
    eid = world.spawn(Position=jnp.array([0.0, 0.0], jnp.float32))
    world.add_systems(step_penalty_reward_system).step({"penalty": 0.5})
    rew = world.get_store("Reward").read(jnp.array([eid]))
    assert jnp.array_equal(rew, jnp.array([-0.5], jnp.float32))


def test_termination_system():
    world = make_world(get_discrete_action_specification)
    eid = world.spawn(Position=jnp.array([11.0, 0.0], jnp.float32))
    world.add_systems(termination_system).step()
    term = world.get_store("Termination").read(jnp.array([eid]))
    assert jnp.array_equal(term, jnp.array([True]))
