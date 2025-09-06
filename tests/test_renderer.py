import numpy as np
from ecsx.envs import build_grid_world
from ecsx.rendering import GridRenderer


def test_renderer_output():
    env = build_grid_world(
        num_agents=1, grid_size=(3, 3), obstacle_positions=[(1, 1)], goal_position=(2, 2)
    )
    renderer = GridRenderer(grid_size=(3, 3), cell_size=1)
    env.reset()
    img = renderer.render(env.world, goal_position=(2, 2))
    assert img.shape == (3, 3, 4)
    blue = np.array([0, 121, 241, 255], np.uint8)
    gray = np.array([130, 130, 130, 255], np.uint8)
    green = np.array([0, 228, 48, 255], np.uint8)
    assert np.any((img == blue).all(axis=-1))
    assert np.any((img == gray).all(axis=-1))
    assert np.any((img == green).all(axis=-1))
