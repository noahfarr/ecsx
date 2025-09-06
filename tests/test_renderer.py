import numpy as np
from ecsx.environments import build_grid_world
from ecsx.rendering import GridRenderer


def test_renderer_output():
    env = build_grid_world(
        num_agents=1,
        grid_size=(3, 3),
        obstacle_positions=[(1, 1)],
        goal_position=(2, 2),
    )
    renderer = GridRenderer(grid_size=(3, 3), cell_size=16)
    env.reset()
    img = renderer.render(env.world, goal_position=(2, 2))
    assert img.shape == (48, 48, 4)

    def center_pixel(cell: tuple[int, int]) -> np.ndarray:
        cx = cell[0] * 16 + 8
        cy = cell[1] * 16 + 8
        return img[cy, cx]

    agent_px = np.array([232, 69, 55, 255], np.uint8)
    obstacle_px = np.array([192, 203, 220, 255], np.uint8)
    goal_px = np.array([234, 165, 108, 255], np.uint8)
    assert (center_pixel((0, 0)) == agent_px).all()
    assert (center_pixel((1, 1)) == obstacle_px).all()
    assert (center_pixel((2, 2)) == goal_px).all()


def test_renderer_returns_owned_array():
    env = build_grid_world(
        num_agents=1,
        grid_size=(3, 3),
        obstacle_positions=[(1, 1)],
        goal_position=(2, 2),
    )
    renderer = GridRenderer(grid_size=(3, 3), cell_size=16)
    env.reset()
    img = renderer.render(env.world, goal_position=(2, 2))
    # The renderer should return an array that owns its memory so it remains valid
    assert img.base is None
