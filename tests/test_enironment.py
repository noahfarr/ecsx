import jax
import jax.numpy as jnp
from ecsx.core.world import World
from ecsx.core.environment import Environment
from ecsx.systems import (
    observation_system,
    discrete_action_system,
    distance_reward_system,
    termination_system,
)
from ecsx.components import (
    get_position_specification,
    get_observation_specification,
    get_discrete_action_specification,
    get_reward_specification,
    get_termination_specification,
)


def test_environment_step_and_reset():
    world = World(capacity=1, key=jax.random.PRNGKey(0))
    world.register_component(get_position_specification())
    world.register_component(get_observation_specification())
    world.register_component(get_discrete_action_specification())
    world.register_component(get_reward_specification())
    world.register_component(get_termination_specification())
    world.spawn(Position=jnp.array([0.0, 0.0], jnp.float32))
    env = Environment(
        world,
        systems=(
            discrete_action_system,
            observation_system,
            distance_reward_system,
            termination_system,
        ),
        action_component="DiscreteAction",
    )
    state, ts = env.reset(jax.random.PRNGKey(1))
    assert jnp.array_equal(ts.observation, jnp.array([0.0, 0.0], jnp.float32))
    state, ts = env.step(jax.random.PRNGKey(2), state, jnp.array(4, jnp.int32))
    assert jnp.array_equal(ts.observation, jnp.array([1.0, 0.0], jnp.float32))
    assert ts.reward == -jnp.linalg.norm(jnp.array([1.0, 0.0], jnp.float32))
    assert bool(ts.done) is False
