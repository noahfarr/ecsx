from dataclasses import dataclass, replace

import jax.numpy as jnp
from jax import tree_util

from ecsx.core.typing import Array, Key, ComponentName, EntityId
from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.component_store import ComponentStore
from ecsx.events.event_specification import EventSpecification
from ecsx.events.event_buffer import EventBuffer


@tree_util.register_pytree_node_class
@dataclass
class WorldState:
    """
    JAX-friendly immustore world state.
    - component_stores: mapping[name] -> ComponentStore
    - alive_mask: (capacity,)
    - random_key: PRNG key
    - time_step: int scalar array
    """

    component_stores: dict[ComponentName, ComponentStore]
    alive_mask: Array
    random_key: Key
    time_step: Array  # int32
    capacity: int
    event_buffers: dict[str, EventBuffer]

    @staticmethod
    def create(capacity: int, key: Key) -> "WorldState":
        return WorldState(
            component_stores={},
            alive_mask=jnp.zeros((capacity,), dtype=bool),
            random_key=key,
            time_step=jnp.array(0, dtype=jnp.int32),
            capacity=capacity,
            event_buffers={},
        )

    def register_component(self, spec: ComponentSpecification) -> "WorldState":
        if spec.name in self.component_stores:
            raise ValueError(f"Component already registered: {spec.name}")
        store = ComponentStore.from_spec(spec, self.capacity)
        new_stores = dict(self.component_stores)
        new_stores[spec.name] = store
        return replace(self, component_stores=new_stores)

    
    def register_event_buffer(self, specification: EventSpecification) -> "WorldState":
        if specification.name in self.event_buffers:
            raise ValueError(f"Event already registered: {specification.name}")
        buf = EventBuffer.from_specification(specification)
        new_bufs = dict(self.event_buffers); new_bufs[specification.name] = buf
        return replace(self, event_buffers=new_bufs)

    def reset_event_buffers(self) -> "WorldState":
        new_bufs = {k: v.clear() for k, v in self.event_buffers.items()}
        return replace(self, event_buffers=new_bufs)

    def write_event_buffer(self, name: str, payloads: jnp.ndarray, count: jnp.ndarray) -> "WorldState":
        buf = self.event_buffers[name].overwrite(payloads, count)
        new_bufs = dict(self.event_buffers); new_bufs[name] = buf
        return replace(self, event_buffers=new_bufs)

    def add_component_to_entity(
        self, name: ComponentName, entity_id: EntityId, value: Array
    ) -> "WorldState":
        store = self._get_store(name)
        value = store.spec.validate_value(value)
        store = store.write(
            jnp.asarray([entity_id], dtype=jnp.int32), value[jnp.newaxis, ...]
        )
        # ensure entity is considered alive if it now has any component
        new_alive = self.alive_mask.at[entity_id].set(True)
        new_stores = dict(self.component_stores)
        new_stores[name] = store
        return replace(self, component_stores=new_stores, alive_mask=new_alive)

    def remove_component_from_entity(
        self, name: ComponentName, entity_id: EntityId
    ) -> "WorldState":
        store = self._get_store(name)
        store = store.clear(jnp.asarray([entity_id], dtype=jnp.int32))
        # Do NOT auto-clear alive here; liveness is controlled by the registry / host sync.
        new_stores = dict(self.component_stores)
        new_stores[name] = store
        return replace(self, component_stores=new_stores)

    def with_alive_mask(self, alive_mask: Array) -> "WorldState":
        if (
            alive_mask.shape != self.alive_mask.shape
            or alive_mask.dtype != self.alive_mask.dtype
        ):
            raise ValueError(
                f"alive_mask shape/dtype mismatch: {alive_mask.shape}, {alive_mask.dtype}"
            )
        return replace(self, alive_mask=alive_mask)

    def tree_flatten(self):
        children = (
            self.component_stores,
            self.alive_mask,
            self.random_key,
            self.time_step,
        )
        aux = self.capacity
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        component_stores, alive_mask, random_key, time_step = children
        return cls(component_stores, alive_mask, random_key, time_step, aux)

    def _get_store(self, name: ComponentName) -> ComponentStore:
        try:
            return self.component_stores[name]
        except KeyError as e:
            raise KeyError(f"Component not registered: {name}") from e

    def _with_store(self, name: ComponentName, store: ComponentStore) -> "WorldState":
        new_stores = dict(self.component_stores)
        new_stores[name] = store
        return replace(self, component_stores=new_stores)
