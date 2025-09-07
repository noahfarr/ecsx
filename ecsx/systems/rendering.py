from typing import Mapping
import numpy as np
import jax.numpy as jnp
from ecsx.core.typing import Array, Key
from ecsx.core.world import WorldState
from ecsx.rendering import render_grid


def _alive_indices(world: WorldState) -> jnp.ndarray:
    try:
        pos_store = world._get_store("Position")
        rend_store = world._get_store("Renderable")
    except KeyError:
        return jnp.array([], jnp.int32)
    mask = world.alive_mask & pos_store.alive_mask & rend_store.alive_mask
    return jnp.where(mask)[0].astype(jnp.int32)


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

    idx = _alive_indices(world)
    if idx.size > 0:
        pos_store = world._get_store("Position")
        rend_store = world._get_store("Renderable")
        positions = np.array(pos_store.read(idx))
        texture_ids = np.array(rend_store.read(idx)).reshape(-1)
    else:
        positions = np.zeros((0, 2))
        texture_ids = np.zeros((0,), dtype=np.int32)

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
