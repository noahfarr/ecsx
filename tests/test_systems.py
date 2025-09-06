import jax
import jax.numpy as jnp
from ecsx.core.world import World
from ecsx.systems import (
    observation_system,
    action_system,
    reward_system,
    termination_system,
)
from ecsx.components import (
    get_position_specification,
    get_observation_specification,
    get_action_specification,
    get_reward_specification,
    get_termination_specification,
)


def make_world():
    world = World(capacity=1, key=jax.random.PRNGKey(0))
    world.register_component(get_position_specification())
    world.register_component(get_observation_specification())
    world.register_component(get_action_specification())
    world.register_component(get_reward_specification())
    world.register_component(get_termination_specification())
    return world


def test_observation_system():
    world = make_world()
    eid = world.spawn(Position=jnp.array([1.0, 2.0], jnp.float32))
    world.add_systems(observation_system).step()
    obs = world.get_store("Observation").read(jnp.array([eid]))
    assert jnp.array_equal(obs, jnp.array([[1.0, 2.0]], jnp.float32))


def test_action_system():
    world = make_world()
    eid = world.spawn(
        Position=jnp.array([0.0, 0.0], jnp.float32),
        Action=jnp.array([1.0, 2.0], jnp.float32),
    )
    world.add_systems(action_system).step()
    pos = world.get_store("Position").read(jnp.array([eid]))
    act = world.get_store("Action").read(jnp.array([eid]))
    assert jnp.array_equal(pos, jnp.array([[1.0, 2.0]], jnp.float32))
    assert jnp.array_equal(act, jnp.array([[0.0, 0.0]], jnp.float32))


def test_reward_system():
    world = make_world()
    eid = world.spawn(Position=jnp.array([3.0, 4.0], jnp.float32))
    world.add_systems(reward_system).step()
    rew = world.get_store("Reward").read(jnp.array([eid]))
    assert jnp.allclose(rew, jnp.array([-5.0], jnp.float32))


def test_termination_system():
    world = make_world()
    eid = world.spawn(Position=jnp.array([11.0, 0.0], jnp.float32))
    world.add_systems(termination_system).step()
    term = world.get_store("Termination").read(jnp.array([eid]))
    assert jnp.array_equal(term, jnp.array([True]))
