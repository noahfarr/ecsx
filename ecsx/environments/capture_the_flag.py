import jax
import jax.numpy as jnp
from typing import Tuple

from ecsx.core.world import World
from ecsx.core.environment import Environment
from ecsx.components import (
    get_position_specification,
    get_observation_specification,
    get_discrete_action_specification,
    get_reward_specification,
    get_termination_specification,
    get_team_specification,
    get_flag_specification,
)
from ecsx.systems import (
    discrete_action_system,
    observation_system,
    flag_capture_system,
    termination_system,
)


def build_capture_the_flag(
    team_size: int = 1,
    map_size: Tuple[int, int] = (10, 10),
    key: jax.Array | None = None,
) -> Environment:
    """Build a simple two-team capture-the-flag environment."""

    rng = key if key is not None else jax.random.PRNGKey(0)
    num_agents = team_size * 2
    capacity = num_agents + 2  # agents plus two flags
    world = World(capacity=capacity, key=rng)

    world.register_component(get_position_specification())
    world.register_component(get_observation_specification())
    world.register_component(get_discrete_action_specification())
    world.register_component(get_reward_specification())
    world.register_component(get_termination_specification())
    world.register_component(get_team_specification())
    world.register_component(get_flag_specification())

    base_positions = jnp.array(
        [[0, 0], [map_size[0] - 1, map_size[1] - 1]], jnp.int32
    )

    agent_ids = []
    for i in range(team_size):
        eid = world.spawn(
            Position=jnp.array([0.0, float(i)], jnp.float32),
            DiscreteAction=jnp.array(0, jnp.int32),
            Team=jnp.array(0, jnp.int32),
        )
        agent_ids.append(int(eid))
    for i in range(team_size):
        eid = world.spawn(
            Position=jnp.array([float(map_size[0] - 1), float(i)], jnp.float32),
            DiscreteAction=jnp.array(0, jnp.int32),
            Team=jnp.array(1, jnp.int32),
        )
        agent_ids.append(int(eid))

    # Flags for each team located at their bases
    world.spawn(
        Position=base_positions[0].astype(jnp.float32),
        Flag=jnp.array([0, 0, -1], jnp.int32),
    )
    world.spawn(
        Position=base_positions[1].astype(jnp.float32),
        Flag=jnp.array([1, 0, -1], jnp.int32),
    )

    systems = [
        discrete_action_system,
        observation_system,
        flag_capture_system,
        termination_system,
    ]

    default_inputs = {
        "grid_size": jnp.array(map_size, jnp.int32),
        "base_positions": base_positions,
    }

    env = Environment(
        world,
        systems=tuple(systems),
        action_component="DiscreteAction",
        agent_indices=agent_ids,
        default_inputs=default_inputs,
    )
    return env

