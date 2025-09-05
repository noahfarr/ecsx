from typing import Iterable

import jax.numpy as jnp

from ..core.world_state import WorldState


def _require_registered(world: WorldState, names: Iterable[str]) -> None:
    for n in names:
        if n not in world.component_stores:
            raise KeyError(f"Component not registered: {n}")


def select(
    world: WorldState, required: tuple[str, ...] = (), forbidden: tuple[str, ...] = ()
) -> jnp.ndarray:
    """
    Build a boolean mask of entities that are alive AND have all required components
    AND do NOT have any forbidden components.
    Returns: mask of shape (capacity,)
    """
    _require_registered(world, required)
    _require_registered(world, forbidden)

    mask = world.alive_mask
    for name in required:
        mask = jnp.logical_and(mask, world.component_stores[name].alive_mask)
    for name in forbidden:
        mask = jnp.logical_and(
            mask, jnp.logical_not(world.component_stores[name].alive_mask)
        )
    return mask


def indices_from_mask(mask: jnp.ndarray) -> jnp.ndarray:
    """Return 1D int indices where mask == True."""
    return jnp.nonzero(mask)[0]
