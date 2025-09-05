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
from ecsx.systems.physics_2d import physics_2d
from ecsx.systems.observation_system import observation_system
from ecsx.systems.action_application_system import action_application_system

from ecsx.algorithms.ppo import PPOConfig, init, train_epoch


def make_world():
    w = World(capacity=8, key=jax.random.PRNGKey(0))
    w.register_component(get_position_specification())
    w.register_component(get_velocity_specification())
    w.register_component(get_observation_specification(4))
    w.register_component(get_action_specification(2))
    w.register_component(get_reward_specification())
    w.register_component(get_termination_specification())
    w.spawn(
        Position=jnp.array([0.0, 0.0], jnp.float32),
        Velocity=jnp.array([0.0, 0.0], jnp.float32),
        Observation=jnp.zeros((4,), jnp.float32),
        Action=jnp.zeros((2,), jnp.float32),
        Reward=jnp.array([0.0], jnp.float32),
        Termination=jnp.array([False]),
    )
    return w

def test_ppo_epoch_runs_without_nan():
    env = EcsxGymnasiumEnvironment(
        world_builder=make_world,
        step_systems=(action_application_system(1.0), physics_2d(0.1), observation_system(True)),
    )
    cfg = PPOConfig(steps_per_rollout=64, minibatch_size=32, update_epochs=2, learning_rate=3e-4)
    state = init(cfg, obs_dim=int(env.observation_space.shape[0]), action_dim=int(env.action_space.shape[0]))
    state, metrics = train_epoch(cfg, state, env)
    assert not any(np.isnan(v) for v in metrics.values())

