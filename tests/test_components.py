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


def test_action_specification():
    spec = get_action_specification()
    assert spec.name == "Action"
    assert spec.shape == (2,)
    assert spec.dtype == jnp.float32
    assert jnp.array_equal(spec.default, jnp.array([0.0, 0.0], jnp.float32))


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
