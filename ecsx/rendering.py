"""Rendering helpers.

This module provides utilities to draw world entities to image buffers using
RayLib.  Entities reference pre-loaded textures through their ``Renderable``
component, and the helpers here allow loading those textures and compositing
them into final RGBA frames.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import jax.numpy as jnp
import raylib
from raylib import ffi


def load_textures(paths: Sequence[str]) -> list:
    """Load a list of texture images from disk."""

    return [raylib.LoadImage(str(p).encode("utf-8")) for p in paths]


def render_grid(
    positions: np.ndarray,
    texture_ids: np.ndarray,
    textures: Sequence,
    grid_size: tuple[int, int],
    cell_size: int = 1,
    background_id: int | None = None,
) -> np.ndarray:
    """Render entities into an RGBA image array.

    Parameters
    ----------
    positions:
        ``(N, 2)`` array of grid coordinates.
    texture_ids:
        ``(N,)`` array of integer indices into ``textures``.
    textures:
        Sequence of RayLib ``Image`` objects.
    grid_size:
        ``(width, height)`` of the grid.
    cell_size:
        Size of a single grid cell in pixels.  Textures are scaled to this size.
    background_id:
        Optional index into ``textures`` used to tile the background. If
        provided, the corresponding image will be drawn into every grid cell
        before rendering entities.
    """
    width = int(grid_size[0]) * cell_size
    height = int(grid_size[1]) * cell_size
    img = raylib.GenImageColor(width, height, raylib.GetColor(0x000000FF))
    img_ptr = ffi.new("Image *", img)

    # Draw background texture on every tile if provided.
    if background_id is not None:
        bg = textures[int(background_id)]
        src_bg = ffi.new("Rectangle*", (0.0, 0.0, float(bg.width), float(bg.height)))[0]
        for gx in range(int(grid_size[0])):
            for gy in range(int(grid_size[1])):
                dst_bg = ffi.new(
                    "Rectangle*",
                    (
                        float(gx * cell_size),
                        float(gy * cell_size),
                        float(cell_size),
                        float(cell_size),
                    ),
                )[0]
                raylib.ImageDraw(
                    img_ptr, bg, src_bg, dst_bg, raylib.GetColor(0xFFFFFFFF)
                )
    for pos, tid in zip(positions, texture_ids):
        x = int(pos[0]) * cell_size
        y = int(pos[1]) * cell_size
        tex = textures[int(tid)]
        src = ffi.new("Rectangle*", (0.0, 0.0, float(tex.width), float(tex.height)))[0]
        dst = ffi.new(
            "Rectangle*", (float(x), float(y), float(cell_size), float(cell_size))
        )[0]
        raylib.ImageDraw(img_ptr, tex, src, dst, raylib.GetColor(0xFFFFFFFF))
    ptr = raylib.LoadImageColors(img_ptr[0])
    flat = ffi.unpack(ptr, width * height)
    data = np.zeros((height, width, 4), dtype=np.uint8)
    for i, px in enumerate(flat):
        data[i // width, i % width] = [px.r, px.g, px.b, px.a]
    raylib.UnloadImageColors(ptr)
    raylib.UnloadImage(img_ptr[0])
    return data


class GridRenderer:
    """Simple renderer wrapper around :func:`render_grid`."""

    def __init__(
        self,
        grid_size: tuple[int, int],
        cell_size: int = 1,
        textures: Sequence | None = None,
        background_id: int | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.textures = textures or []
        self.background_id = background_id

    def render(self, world, **inputs) -> np.ndarray:
        textures = inputs.get("textures", self.textures)
        background_id = inputs.get("background_id", self.background_id)
        pos_store = world.get_store("Position")
        rend_store = world.get_store("Renderable")
        mask = (
            np.array(world.state.alive_mask)
            & np.array(pos_store.alive_mask)
            & np.array(rend_store.alive_mask)
        )
        idx = jnp.where(mask)[0].astype(jnp.int32)
        if idx.size == 0:
            width = int(self.grid_size[0]) * self.cell_size
            height = int(self.grid_size[1]) * self.cell_size
            return np.zeros((height, width, 4), dtype=np.uint8)
        positions = np.array(pos_store.read(idx))
        texture_ids = np.array(rend_store.read(idx)).reshape(-1)
        return render_grid(
            positions,
            texture_ids,
            textures,
            self.grid_size,
            self.cell_size,
            background_id,
        )
