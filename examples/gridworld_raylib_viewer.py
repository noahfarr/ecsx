"""
Interactive Gridworld viewer using raylib (pyray binding).

Install prerequisites:
  - Linux: apt-get install libraylib-dev (or build from source)
  - Python: pip install pyray   (package name may be 'pyray')

Run:
  python examples/gridworld_raylib_viewer.py
Controls:
  Arrow keys = move agent (Up/Right/Down/Left), Space = no-op, Esc/Close to exit.
"""
import sys
try:
    import pyray as rl
except Exception as e:
    print("Raylib (pyray) not available. Install with `pip install pyray` and system libs.\n", e)
    sys.exit(0)

import jax
import jax.numpy as jnp

from ecsx.world.world_api import World
from ecsx.systems.pipeline_builder import build_step

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


GRID_WIDTH, GRID_HEIGHT = 16, 12
CELL_SIZE = 40
WINDOW_W, WINDOW_H = GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE

# pleasant pastel color palette
BG_COLOR_A = rl.Color(236, 240, 241, 255)
BG_COLOR_B = rl.Color(220, 223, 225, 255)
GRID_LINE_COLOR = rl.Color(189, 195, 199, 255)
OBSTACLE_COLOR = rl.DARKGRAY
AGENT_COLOR = rl.RED
GOAL_COLOR = rl.GOLD

def build_world() -> World:
    w = World(capacity=256, key=jax.random.PRNGKey(0))
    w.register_component(get_grid_position_specification())
    w.register_component(get_discrete_action_specification(5))
    w.register_component(get_grid_goal_specification())
    w.register_component(get_grid_obstacle_specification())
    w.register_component(get_observation_specification(4))  # [ax/W, ay/H, gx/W, gy/H]
    w.register_component(get_reward_specification())
    w.register_component(get_termination_specification())

    # Agent at (1,1)
    agent = w.spawn(GridPosition=jnp.array([1, 1], jnp.int32),
                    DiscreteAction=jnp.array([0], jnp.int32),
                    Observation=jnp.zeros((4,), jnp.float32),
                    Reward=jnp.array([0.0], jnp.float32),
                    Termination=jnp.array([False]))
    # Goal at (10,6)
    w.spawn(GridGoal=jnp.array(True), GridPosition=jnp.array([10, 6], jnp.int32))
    # Obstacles (simple wall)
    for x in range(3, 13):
        w.spawn(GridObstacle=jnp.array(True), GridPosition=jnp.array([x, 4], jnp.int32))

    return w

def systems():
    return (
        grid_action_system(GRID_WIDTH, GRID_HEIGHT),
        grid_observation_system(GRID_WIDTH, GRID_HEIGHT),
        grid_reward_system(1.0),
    )

def main():
    world = build_world()
    step_fn = build_step(systems())

    rl.init_window(WINDOW_W, WINDOW_H, "ecsx gridworld (raylib)")
    rl.set_target_fps(30)

    while not rl.window_should_close():
        # Handle input -> DiscreteAction
        action = 0  # no-op
        if rl.is_key_pressed(rl.KEY_UP):    action = 1
        if rl.is_key_pressed(rl.KEY_RIGHT): action = 2
        if rl.is_key_pressed(rl.KEY_DOWN):  action = 3
        if rl.is_key_pressed(rl.KEY_LEFT):  action = 4
        # write action to the first agent (the first with DiscreteAction)
        mask = (world.state.component_stores["DiscreteAction"].alive_mask &
                world.state.component_stores["GridPosition"].alive_mask &
                world.state.alive_mask)
        agent_idx = jnp.nonzero(mask)[0]
        if agent_idx.size > 0:
            act_store = world.state.component_stores["DiscreteAction"]
            act_store = act_store.write(jnp.asarray([int(agent_idx[0])]),
                                        jnp.asarray([[action]], jnp.int32))
            world._world = world.state._with_store("DiscreteAction", act_store)

        world._world = step_fn(world.state, {})

        # draw
        rl.begin_drawing()
        rl.clear_background(BG_COLOR_A)

        # cell background pattern
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                col = BG_COLOR_A if (x + y) % 2 == 0 else BG_COLOR_B
                rl.draw_rectangle(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE, col)

        # grid lines
        for x in range(GRID_WIDTH + 1):
            rl.draw_line(x * CELL_SIZE, 0, x * CELL_SIZE, WINDOW_H, GRID_LINE_COLOR)
        for y in range(GRID_HEIGHT + 1):
            rl.draw_line(0, y * CELL_SIZE, WINDOW_W, y * CELL_SIZE, GRID_LINE_COLOR)

        # obstacles
        omask = (world.state.component_stores["GridObstacle"].alive_mask &
                 world.state.component_stores["GridPosition"].alive_mask &
                 world.state.alive_mask)
        oidx = jnp.nonzero(omask)[0]
        if int(oidx.size) > 0:
            opos = world.state.component_stores["GridPosition"].read(oidx)
            for i in range(int(oidx.size)):
                x, y = int(opos[i, 0]), int(opos[i, 1])
                rl.draw_rectangle(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE, OBSTACLE_COLOR)

        # goals
        gmask = (world.state.component_stores["GridGoal"].alive_mask &
                 world.state.component_stores["GridPosition"].alive_mask &
                 world.state.alive_mask)
        gidx = jnp.nonzero(gmask)[0]
        if int(gidx.size) > 0:
            gpos = world.state.component_stores["GridPosition"].read(gidx)
            for i in range(int(gidx.size)):
                x, y = int(gpos[i, 0]), int(gpos[i, 1])
                cx = x * CELL_SIZE + CELL_SIZE // 2
                cy = y * CELL_SIZE + CELL_SIZE // 2
                radius = CELL_SIZE // 3
                rl.draw_circle(cx, cy, radius, GOAL_COLOR)
                rl.draw_circle_lines(cx, cy, radius, rl.BLACK)

        # agents
        amask = (world.state.component_stores["DiscreteAction"].alive_mask &
                 world.state.component_stores["GridPosition"].alive_mask &
                 world.state.alive_mask)
        aidx = jnp.nonzero(amask)[0]
        if int(aidx.size) > 0:
            apos = world.state.component_stores["GridPosition"].read(aidx)
            for i in range(int(aidx.size)):
                x, y = int(apos[i, 0]), int(apos[i, 1])
                cx = x * CELL_SIZE + CELL_SIZE // 2
                cy = y * CELL_SIZE + CELL_SIZE // 2
                radius = int(CELL_SIZE * 0.35)
                rl.draw_circle(cx, cy, radius, AGENT_COLOR)
                rl.draw_circle_lines(cx, cy, radius, rl.BLACK)

        rl.end_drawing()

    rl.close_window()

if __name__ == "__main__":
    main()

