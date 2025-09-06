from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pyray as pr
from pathlib import Path

from ecsx.core.world import World


class GridRenderer:
    """Render grid worlds to numpy arrays using raylib with sprites."""

    def __init__(
        self,
        grid_size: tuple[int, int],
        cell_size: int = 16,
        agent_sprite: str | None = None,
        obstacle_sprite: str | None = None,
        goal_sprite: str | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.cell_size = cell_size

        asset_dir = Path(__file__).with_suffix("").parent / "assets"
        self.agent_img = pr.load_image(agent_sprite or str(asset_dir / "agent.png"))
        self.obstacle_img = pr.load_image(
            obstacle_sprite or str(asset_dir / "obstacle.png")
        )
        goal_path = goal_sprite or str(asset_dir / "goal.png")
        self.goal_img = pr.load_image(goal_path) if Path(goal_path).exists() else None

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
            sprite = self.obstacle_img if obs else self.agent_img
            pr.image_draw(
                img,
                sprite,
                pr.Rectangle(0, 0, sprite.width, sprite.height),
                pr.Rectangle(x, y, self.cell_size, self.cell_size),
                pr.WHITE,
            )

        if goal_position is not None:
            gx = int(goal_position[0]) * self.cell_size
            gy = int(goal_position[1]) * self.cell_size
            sprite = self.goal_img if self.goal_img is not None else self.agent_img
            pr.image_draw(
                img,
                sprite,
                pr.Rectangle(0, 0, sprite.width, sprite.height),
                pr.Rectangle(gx, gy, self.cell_size, self.cell_size),
                pr.WHITE,
            )

        colors = pr.load_image_colors(img)
        buf = pr.ffi.buffer(colors, width * height * 4)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4)).copy()
        pr.unload_image_colors(colors)
        pr.unload_image(img)
        return arr

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        for img in [self.agent_img, self.obstacle_img, self.goal_img]:
            try:
                if img is not None:
                    pr.unload_image(img)
            except Exception:
                pass
