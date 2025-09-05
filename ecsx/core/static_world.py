from dataclasses import dataclass, replace
from typing import Tuple, Dict, Iterator, Mapping
import jax.numpy as jnp
from jax import tree_util

from .component_store import ComponentStore
from .world_state import WorldState
from ..events.event_buffer import EventBuffer


@dataclass(frozen=True)
class StaticWorld:
    """
    Immutable, jit-friendly view over WorldState:
      - store_names: fixed order of component stores (hashable metadata)
      - stores: tuple aligned to store_names
      - alive_mask, random_key, time_step, capacity, event_buffers: same as WorldState
    Provides a minimal compat surface: component_stores[...] (read-only view) and _with_store(name, store).
    """
    store_names: Tuple[str, ...]
    stores: Tuple[ComponentStore, ...]
    alive_mask: jnp.ndarray
    random_key: jnp.ndarray
    time_step: jnp.ndarray
    capacity: int
    event_buffers: Dict[str, EventBuffer]

    @staticmethod
    def freeze(world: WorldState, name_order: Tuple[str, ...]) -> "StaticWorld":
        stores = tuple(world.component_stores[n] for n in name_order)
        return StaticWorld(
            store_names=tuple(name_order),
            stores=stores,
            alive_mask=world.alive_mask,
            random_key=world.random_key,
            time_step=world.time_step,
            capacity=world.capacity,
            event_buffers=world.event_buffers,
        )

    def thaw(self) -> WorldState:
        d = {n: s for n, s in zip(self.store_names, self.stores)}
        base = WorldState.create(self.capacity, self.random_key)
        return replace(
            base,
            component_stores=d,
            alive_mask=self.alive_mask,
            random_key=self.random_key,
            time_step=self.time_step,
            event_buffers=self.event_buffers,
        )

    class StoreMapping(Mapping[str, ComponentStore]):
        def __init__(self, names: Tuple[str, ...], stores: Tuple[ComponentStore, ...]):
            self._names = names
            self._stores = stores
            # Fast Python map resolved at trace time; not part of PyTree.
            self._name_to_idx = {n: i for i, n in enumerate(names)}
        def __getitem__(self, key: str) -> ComponentStore:
            return self._stores[self._name_to_idx[key]]
        def __iter__(self) -> Iterator[str]:
            return iter(self._names)
        def __len__(self) -> int:
            return len(self._names)

    @property
    def component_stores(self) -> Mapping[str, ComponentStore]:
        # Read-only mapping view (resolved at trace time).
        return StaticWorld.StoreMapping(self.store_names, self.stores)

    def _with_store(self, name: str, store: ComponentStore) -> "StaticWorld":
        idx = {n: i for i, n in enumerate(self.store_names)}[name]
        new_stores = self.stores[:idx] + (store,) + self.stores[idx + 1 :]
        return replace(self, stores=new_stores)

    # Event buffer ops (used by pipeline/systems)
    def reset_event_buffers(self) -> "StaticWorld":
        new_bufs = {k: v.clear() for k, v in self.event_buffers.items()}
        return replace(self, event_buffers=new_bufs)

    def write_event_buffer(self, name: str, payloads: jnp.ndarray, count: jnp.ndarray) -> "StaticWorld":
        buf = self.event_buffers[name].overwrite(payloads, count)
        new_bufs = dict(self.event_buffers); new_bufs[name] = buf
        return replace(self, event_buffers=new_bufs)


# PyTree registration (metadata must be hashable/array-free)
tree_util.register_pytree_node(
    StaticWorld,
    lambda w: (
        (w.stores, w.alive_mask, w.random_key, w.time_step, w.event_buffers),
        (w.store_names, w.capacity),
    ),
    lambda aux, kids: StaticWorld(
        store_names=aux[0],
        stores=kids[0],
        alive_mask=kids[1],
        random_key=kids[2],
        time_step=kids[3],
        capacity=aux[1],
        event_buffers=kids[4],
    ),
)

