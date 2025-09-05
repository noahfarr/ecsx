from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import jax.numpy as jnp

from ..core.world_state import WorldState
from ..core.component_specification import ComponentSpecification
from ..core.component_store import ComponentStore
from ..events.event_specification import EventSpecification
from ..events.event_buffer import EventBuffer


def _dtype_to_str(dt) -> str:
    return np.dtype(dt).str  # e.g. '<f4', '<i4', '|b1'


def _str_to_jnp_dtype(s: str):
    return jnp.dtype(np.dtype(s))


def save_world_state(path: str | Path, world: WorldState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten component stores
    store_names = list(world.component_stores.keys())
    store_specs = []
    store_data = {}
    for name, store in world.component_stores.items():
        spec = store.specification
        store_specs.append((name, spec.shape, _dtype_to_str(spec.dtype), np.asarray(spec.default_value)))
        store_data[f"{name}__data"] = np.asarray(store.data)
        store_data[f"{name}__mask"] = np.asarray(store.present_mask)

    # Flatten event buffers
    evt_names = list(world.event_buffers.keys())
    evt_specs = []
    evt_data = {}
    for name, buf in world.event_buffers.items():
        spec = buf.specification
        evt_specs.append((name, spec.shape, _dtype_to_str(spec.dtype), int(spec.capacity)))
        evt_data[f"{name}__data"] = np.asarray(buf.data)
        evt_data[f"{name}__count"] = np.asarray(buf.count)

    np.savez_compressed(
        path,
        capacity=np.array(world.capacity, dtype=np.int32),
        time_step=np.asarray(world.time_step),
        random_key=np.asarray(world.random_key),
        alive_mask=np.asarray(world.alive_mask),
        store_names=np.array(store_names, dtype=object),
        store_specs=np.array(store_specs, dtype=object),
        evt_names=np.array(evt_names, dtype=object),
        evt_specs=np.array(evt_specs, dtype=object),
        **store_data,
        **evt_data,
    )


def load_world_state(path: str | Path) -> WorldState:
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        capacity = int(z["capacity"])
        world = WorldState.create(capacity=capacity, key=jnp.asarray(z["random_key"]))
        world = replace(world,
                        time_step=jnp.asarray(z["time_step"]),
                        alive_mask=jnp.asarray(z["alive_mask"]))

        # Stores
        store_names = list(z["store_names"])
        for name, shape, dstr, default in z["store_specs"]:
            spec = ComponentSpecification(str(name), tuple(shape.tolist()), _str_to_jnp_dtype(str(dstr)), jnp.asarray(default))
            world = world.register_component(spec)
            data = jnp.asarray(z[f"{name}__data"])
            mask = jnp.asarray(z[f"{name}__mask"]).astype(bool)
            store = world.component_stores[str(name)]
            store = ComponentStore(spec, data, mask)
            world = world._with_store(str(name), store)

        # Events
        evt_names = list(z["evt_names"])
        for name, shape, dstr, cap in z["evt_specs"]:
            spec = EventSpecification(str(name), tuple(shape.tolist()), _str_to_jnp_dtype(str(dstr)), int(cap))
            world = world.register_event_buffer(spec)
            data = jnp.asarray(z[f"{name}__data"])
            count = jnp.asarray(z[f"{name}__count"]).astype(jnp.int32)
            buf = EventBuffer(spec, data, count)
            world = replace(world, event_buffers={**world.event_buffers, str(name): buf})

        return world

