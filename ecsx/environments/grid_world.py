import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Sequence, Tuple
from ecsx.core.world import World
from ecsx.core.environment import Environment
from ecsx.core.event_specification import EventSpecification
from ecsx.components import (
    get_position_specification,
    get_observation_specification,
    get_discrete_action_specification,
    get_reward_specification,
    get_termination_specification,
    get_obstacle_specification,
    get_renderable_specification,
)
from ecsx.systems import (
    discrete_action_system,
    observation_system,
    grid_observation_system,
    goal_reward_system,
    step_penalty_reward_system,
    termination_system,
    render_system,
)
from ecsx.rendering import load_textures


def _generate_obstacles(
    key: jax.Array,
    num_obstacles: int,
    grid_size: Tuple[int, int],
    num_agents: int,
    goal_position: Tuple[int, int] | None,
) -> Sequence[Tuple[int, int]]:
    """Sample obstacle positions avoiding agents and optional goal."""

    all_positions = [(x, y) for x in range(grid_size[0]) for y in range(grid_size[1])]
    occupied = {(0, i) for i in range(num_agents)}
    if goal_position is not None:
        occupied.add(goal_position)
    candidates = [pos for pos in all_positions if pos not in occupied]
    if num_obstacles > len(candidates):
        raise ValueError("Not enough free cells for requested obstacles")
    cand_arr = jnp.array(candidates, jnp.int32)
    idx = jax.random.choice(key, cand_arr.shape[0], (num_obstacles,), replace=False)
    sampled = cand_arr[idx]
    return [tuple(map(int, p)) for p in sampled]


def build_grid_world(
    num_agents: int = 1,
    grid_size: Tuple[int, int] = (5, 5),
    obstacle_positions: Sequence[Tuple[int, int]] | None = None,
    num_obstacles: int = 0,
    goal_position: Tuple[int, int] | None = None,
    view_radius: int | None = None,
    step_penalty: float | None = None,
    key: jax.Array | None = None,
) -> Environment:
    """Build a grid world environment.

    Parameters
    ----------
    num_agents: number of controllable agents.
    grid_size: width and height of the grid.
    obstacle_positions: coordinates of static obstacles.
    num_obstacles: number of procedurally generated obstacles if ``obstacle_positions`` is None.
    goal_position: optional goal coordinate for rewards.
    view_radius: if provided, agents receive a square local observation with this radius.
    step_penalty: constant penalty per step.
    key: optional random key for world initialization.
    """

    rng = key if key is not None else jax.random.PRNGKey(0)
    if obstacle_positions is None and num_obstacles > 0:
        rng, obst_key = jax.random.split(rng)
        obstacle_positions = _generate_obstacles(
            obst_key, num_obstacles, grid_size, num_agents, goal_position
        )
    capacity = num_agents + (len(obstacle_positions) if obstacle_positions else 0)
    world = World(capacity=capacity, key=rng)

    if view_radius is None:
        obs_shape = (2,)
        obs_dtype = jnp.float32
    else:
        obs_shape = ((2 * view_radius + 1) ** 2,)
        obs_dtype = jnp.int32

    world.register_component(get_position_specification())
    world.register_component(
        get_observation_specification(shape=obs_shape, dtype=obs_dtype)
    )
    world.register_component(get_discrete_action_specification())
    world.register_component(get_reward_specification())
    world.register_component(get_termination_specification())
    world.register_component(get_obstacle_specification())
    world.register_component(get_renderable_specification())

    frame_spec = EventSpecification(
        "Frame",
        (grid_size[1], grid_size[0], 4),
        jnp.uint8,
        1,
    )
    world._world = world.state.register_event_buffer(frame_spec)

    asset_dir = Path(__file__).resolve().parents[2] / "assets"
    textures = load_textures(
        [
            asset_dir / "agent.png",
            asset_dir / "obstacle.png",
            asset_dir / "goal.png",
            asset_dir / "floor.png",
        ]
    )

    agent_ids = []
    for i in range(num_agents):
        eid = world.spawn(
            Position=jnp.array([0.0, float(i)], jnp.float32),
            DiscreteAction=jnp.array(0, jnp.int32),
            Renderable=jnp.array(0, jnp.int32),
        )
        agent_ids.append(int(eid))

    if obstacle_positions:
        for pos in obstacle_positions:
            world.spawn(
                Position=jnp.array(pos, jnp.float32),
                Obstacle=jnp.array(True),
                Renderable=jnp.array(1, jnp.int32),
            )

    systems = [discrete_action_system]
    if view_radius is None:
        systems.append(observation_system)
    else:
        systems.append(grid_observation_system)
    if goal_position is not None:
        systems.append(goal_reward_system)
    if step_penalty is not None:
        systems.append(step_penalty_reward_system)
    systems.append(termination_system)
    systems.append(render_system)

    default_inputs = {
        "grid_size": jnp.array(grid_size, jnp.int32),
        "textures": textures,
    }
    if goal_position is not None:
        default_inputs["goal_position"] = jnp.array(goal_position, jnp.int32)
    if view_radius is not None:
        default_inputs["view_radius"] = view_radius
    if step_penalty is not None:
        default_inputs["penalty"] = step_penalty

    env = Environment(
        world,
        systems=tuple(systems),
        action_component="DiscreteAction",
        agent_indices=agent_ids,
        default_inputs=default_inputs,
    )
    return env
