import jax.numpy as jnp
from ecsx.components import *


def test_position_specification():
    spec = get_position_specification()
    assert spec.name == "Position"
    assert spec.shape == (2,)
    assert spec.dtype == jnp.float32
    assert jnp.array_equal(spec.default, jnp.array([0.0, 0.0], jnp.float32))


def test_velocity_specification():
    spec = get_velocity_specification()
    assert spec.name == "Velocity"
    assert spec.shape == (2,)
    assert spec.dtype == jnp.float32
    assert jnp.array_equal(spec.default, jnp.array([0.0, 0.0], jnp.float32))


def test_observation_specification():
    spec = get_observation_specification()
    assert spec.name == "Observation"
    assert spec.shape == (2,)
    assert spec.dtype == jnp.float32
    assert jnp.array_equal(spec.default, jnp.array([0.0, 0.0], jnp.float32))


def test_continuous_action_specification():
    spec = get_continuous_action_specification()
    assert spec.name == "ContinuousAction"
    assert spec.shape == (2,)
    assert spec.dtype == jnp.float32
    assert jnp.array_equal(spec.default, jnp.array([0.0, 0.0], jnp.float32))


def test_discrete_action_specification():
    spec = get_discrete_action_specification()
    assert spec.name == "DiscreteAction"
    assert spec.shape == ()
    assert spec.dtype == jnp.int32
    assert jnp.array_equal(spec.default, jnp.array(0, jnp.int32))


def test_obstacle_specification():
    spec = get_obstacle_specification()
    assert spec.name == "Obstacle"
    assert spec.shape == ()
    assert spec.dtype == jnp.bool_
    assert jnp.array_equal(spec.default, jnp.array(False, jnp.bool_))


def test_reward_specification():
    spec = get_reward_specification()
    assert spec.name == "Reward"
    assert spec.shape == ()
    assert spec.dtype == jnp.float32
    assert jnp.array_equal(spec.default, jnp.array(0.0, jnp.float32))


def test_termination_specification():
    spec = get_termination_specification()
    assert spec.name == "Termination"
    assert spec.shape == ()
    assert spec.dtype == jnp.bool_
    assert jnp.array_equal(spec.default, jnp.array(False, jnp.bool_))


def test_renderable_specification():
    spec = get_renderable_specification()
    assert spec.name == "Renderable"
    assert spec.shape == ()
    assert spec.dtype == jnp.int32
    assert jnp.array_equal(spec.default, jnp.array(0, jnp.int32))
