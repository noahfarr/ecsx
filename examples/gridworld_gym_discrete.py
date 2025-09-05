import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp

from ecsx.world.world_api import World
from ecsx.integration.gymnasium_discrete_bridge import EcsxGymnasiumDiscreteEnvironment

from ecsx.components.grid_position import get_grid_position_specification
from ecsx.components.discrete_action import get_discrete_action_specification
from ecsx.components.grid_goal import get_grid_goal_specification
from ecsx.components.grid_obstacle import get_grid_obstacle_specification
from ecsx.components.observation import get_observation_specification
from ecsx.components.reward import get_reward_specification
from ecsx.components.termination import get_termination_specification

from ecsx.systems.grid_action_system import grid_action_system
from ecsx.systems.grid_observation_system import grid_observation_system
from ecsx.systems.grid_reward_system import grid_reward_system

W, H = 10, 8

def make_world():
    w = World(capacity=128, key=jax.random.PRNGKey(0))
    w.register_component(get_grid_position_specification())
    w.register_component(get_discrete_action_specification(5))
    w.register_component(get_grid_goal_specification())
    w.register_component(get_grid_obstacle_specification())
    w.register_component(get_observation_specification(4))
    w.register_component(get_reward_specification())
    w.register_component(get_termination_specification())
    agent = w.spawn(GridPosition=jnp.array([1,1], jnp.int32),
                    DiscreteAction=jnp.array([0], jnp.int32),
                    Observation=jnp.zeros((4,), jnp.float32),
                    Reward=jnp.array([0.0], jnp.float32),
                    Termination=jnp.array([False]))
    w.spawn(GridGoal=jnp.array(True), GridPosition=jnp.array([W-2, H-2], jnp.int32))
    for x in range(2, W-2):
        w.spawn(GridObstacle=jnp.array(True), GridPosition=jnp.array([x, H//2], jnp.int32))
    return w

env = EcsxGymnasiumDiscreteEnvironment(
    world_builder=make_world,
    step_systems=(grid_action_system(W, H),
                  grid_observation_system(W, H),
                  grid_reward_system(1.0)),
    num_actions=5,
)

obs, info = env.reset()
done = False
total = 0.0
while not done:
    action = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(action)
    total += rew
    done = term or trunc
print("Episode return:", total)

