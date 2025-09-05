from typing import Iterable

import jax.numpy as jnp

from ..core.world_state import WorldState
from ..core.static_world import StaticWorld

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
    if isinstance(world, StaticWorld):
        name_to_idx = {n: i for i, n in enumerate(world.store_names)}
        mask = world.alive_mask
        for n in required:
            mask = jnp.logical_and(mask, world.stores[name_to_idx[n]].alive_mask)
        for n in forbidden:
            mask = jnp.logical_and(mask, jnp.logical_not(world.stores[name_to_idx[n]].alive_mask))
        return mask

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

def select_static(sw: StaticWorld,
                  required: tuple[str, ...] = (),
                  forbidden: tuple[str, ...] = ()) -> jnp.ndarray:
    """Compute mask on StaticWorld without any dict lookups (JIT-friendly)."""
    name_to_idx = {n: i for i, n in enumerate(sw.store_names)}

    mask = sw.alive_mask
    for name in required:
        i = name_to_idx[name]
        mask = jnp.logical_and(mask, sw.stores[i].alive_mask)
    for name in forbidden:
        i = name_to_idx[name]
        mask = jnp.logical_and(mask, jnp.logical_not(sw.stores[i].alive_mask))
    return mask


def compile_selector_for_static(required: tuple[str, ...],
                                forbidden: tuple[str, ...],
                                name_order: tuple[str, ...]):
    """
    Pre-resolve component names to store indices. Returns fn(StaticWorld)->mask.
    Use when building a static, jitted step where name_order is fixed.
    """
    import jax.numpy as jnp  # local to avoid circulars in some setups
    idx = {n: i for i, n in enumerate(name_order)}
    req_idx = tuple(idx[n] for n in required)
    forb_idx = tuple(idx[n] for n in forbidden)

    def _fn(sw: StaticWorld):
        mask = sw.alive_mask
        for i in req_idx:
            mask = jnp.logical_and(mask, sw.stores[i].alive_mask)
        for i in forb_idx:
            mask = jnp.logical_and(mask, jnp.logical_not(sw.stores[i].alive_mask))
        return mask

    return _fn

