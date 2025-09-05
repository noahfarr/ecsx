import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from ecsx.world.world_api import World
from ecsx.integration.gymnasium_bridge import EcsxGymnasiumEnvironment
from ecsx.components.position import get_position_specification
from ecsx.components.velocity import get_velocity_specification
from ecsx.components.observation import get_observation_specification
from ecsx.components.action import get_action_specification
from ecsx.components.reward import get_reward_specification
from ecsx.components.termination import get_termination_specification

from ecsx.systems.pipeline_builder import build_step
from ecsx.systems.physics_2d import physics_2d
from ecsx.systems.observation_system import observation_system
from ecsx.systems.action_application_system import action_application_system
from ecsx.query.query_engine import select


def make_world():
    w = World(capacity=8, key=jax.random.PRNGKey(0))
    w.register_component(get_position_specification())
    w.register_component(get_velocity_specification())
    w.register_component(get_observation_specification(4))  # pos(2)+vel(2)
    w.register_component(get_action_specification(2))
    w.register_component(get_reward_specification())
    w.register_component(get_termination_specification())
    # spawn single agent
    agent = w.spawn(
        Position=jnp.array([0.0, 0.0], jnp.float32),
        Velocity=jnp.array([0.0, 0.0], jnp.float32),
        Observation=jnp.zeros((4,), jnp.float32),
        Action=jnp.zeros((2,), jnp.float32),
        Reward=jnp.array([0.0], jnp.float32),
        Termination=jnp.array([False]),
    )
    return w

# simple per-step reward +1 system for all with Reward
def reward_plus_one_system(world, key, inputs):
    mask = select(world, required=("Reward",))
    idx = jnp.nonzero(mask, size=world.capacity, fill_value=-1)[0]
    cnt = jnp.sum(mask)
    if int(cnt) == 0:
        return world
    rew = world.component_stores["Reward"].read(idx[:cnt])
    rew = rew.at[:, 0].add(1.0)
    store = world.component_stores["Reward"].write(idx[:cnt], rew)
    return world._with_store("Reward", store)

# terminate once x position > 0.5
def termination_by_position(world, key, inputs):
    mask = select(world, required=("Position", "Termination"))
    idx = jnp.nonzero(mask, size=world.capacity, fill_value=-1)[0]
    cnt = jnp.sum(mask)
    if int(cnt) == 0:
        return world
    pos = world.component_stores["Position"].read(idx[:cnt])
    term = (pos[:, 0] > 0.5).astype(jnp.bool_).reshape((-1, 1))
    store = world.component_stores["Termination"].write(idx[:cnt], term)
    return world._with_store("Termination", store)

def test_gym_wrapper_runs_and_shapes():
    env = EcsxGymnasiumEnvironment(
        world_builder=make_world,
        step_systems=(action_application_system(velocity_scale=1.0),
                      physics_2d(0.5),
                      observation_system(True),
                      reward_plus_one_system,
                      termination_by_position),
    )
    obs, info = env.reset()
    assert obs.shape == (4,)
    # action sets velocity -> moves x by 0.5*action
    action = np.array([0.4, 0.0], dtype=np.float32)
    obs, rew, term, trunc, info = env.step(action)
    # after first step: position x = 0.2, velocity x = 0.4; obs = [0.2, 0, 0.4, 0]
    assert np.allclose(obs[:2], np.array([0.2, 0.0], np.float32))
    assert rew == 1.0
    assert term is False and trunc is False

    # keep stepping until termination
    steps = 0
    while True:
        obs, rew, term, trunc, info = env.step(action)
        steps += 1
        if term:
            break
    assert steps >= 1
    # reward is per-step and reset by wrapper
    assert rew == 1.0

