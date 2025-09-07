from typing import Mapping
import numpy as np
import jax.numpy as jnp
from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState
from ecsx.rendering import render_grid


def render_system(world: WorldState, key: Key, inputs: Mapping[str, Array]) -> WorldState:
    """Produce a rendered image of the current world state.

    The resulting image is written to the ``Frame`` event buffer if it exists.
    ``inputs`` must contain ``grid_size`` and ``textures`` and may optionally
    specify ``cell_size`` and ``background_id``.
    """
    if "Frame" not in world.event_buffers:
        return world
    grid_size = inputs.get("grid_size")
    textures = inputs.get("textures")
    if grid_size is None or textures is None:
        return world
    grid_size = tuple(int(x) for x in np.array(grid_size))
    cell_size = int(inputs.get("cell_size", 1))
    background_id = inputs.get("background_id")

    pos_store = world._get_store("Position")
    rend_store = world._get_store("Renderable")
    idx = jnp.arange(world.alive_mask.shape[0], dtype=jnp.int32)
    mask = np.array(
        world.alive_mask & pos_store.alive_mask & rend_store.alive_mask
    )
    positions = np.array(pos_store.read(idx))[mask]
    texture_ids = np.array(rend_store.read(idx).reshape(-1))[mask]

    image = render_grid(
        positions,
        texture_ids,
        textures,
        grid_size,
        cell_size,
        background_id,
    )
    frame = jnp.asarray(image)
    payloads = jnp.zeros((1, *frame.shape), dtype=jnp.uint8)
    payloads = payloads.at[0].set(frame)
    world = world.write_event_buffer("Frame", payloads, jnp.array(1, jnp.int32))
    return world
