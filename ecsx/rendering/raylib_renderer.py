from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pyray as pr

from ecsx.core.world import World


class GridRenderer:
    """Render grid worlds to numpy arrays using raylib with solid colors."""

    def __init__(self, grid_size: tuple[int, int], cell_size: int = 16) -> None:
        self.grid_size = grid_size
        self.cell_size = cell_size

    def render(
        self, world: World, goal_position: tuple[int, int] | None = None
    ) -> np.ndarray:
        width = self.grid_size[0] * self.cell_size
        height = self.grid_size[1] * self.cell_size
        img = pr.gen_image_color(width, height, pr.RAYWHITE)

        pos_store = world.get_store("Position")
        idx = jnp.where(world.state.alive_mask)[0]
        positions = np.array(pos_store.read(idx)).astype(int)
        try:
            obst_store = world.get_store("Obstacle")
            is_obstacle = np.array(obst_store.read(idx)).astype(bool)
        except KeyError:
            is_obstacle = np.zeros(len(idx), dtype=bool)

        for p, obs in zip(positions, is_obstacle):
            x = int(p[0]) * self.cell_size
            y = int(p[1]) * self.cell_size
            color = pr.GRAY if obs else pr.BLUE
            pr.image_draw_rectangle(img, x, y, self.cell_size, self.cell_size, color)

        if goal_position is not None:
            gx = int(goal_position[0]) * self.cell_size
            gy = int(goal_position[1]) * self.cell_size
            pr.image_draw_rectangle(img, gx, gy, self.cell_size, self.cell_size, pr.GREEN)

        colors = pr.load_image_colors(img)
        buf = pr.ffi.buffer(colors, width * height * 4)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
        pr.unload_image_colors(colors)
        pr.unload_image(img)
        return arr
