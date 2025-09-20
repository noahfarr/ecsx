from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
from jax import tree_util as jtu, tree_util

from ecsx.core.typing import Array, Key, ComponentName, EntityId
from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.component_store import ComponentStore
from ecsx.core.entity_registry import EntityRegistry
from ecsx.core.event_specification import EventSpecification
from ecsx.core.event_buffer import EventBuffer
from ecsx.core.system import System, build_step


@tree_util.register_pytree_node_class
@dataclass
class WorldState:
    component_stores: dict[ComponentName, ComponentStore]
    alive_mask: Array
    random_key: Key
    time_step: Array
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

    def _add_batch_dim(self, value: Any) -> Any:
        return jtu.tree_map(lambda leaf: jnp.asarray(leaf)[jnp.newaxis, ...], value)

    def add_component_to_entity(
        self, name: ComponentName, entity_id: EntityId, value: Any
    ) -> "WorldState":
        store = self._get_store(name)
        batched = self._add_batch_dim(value)
        store = store.write(jnp.asarray([entity_id], dtype=jnp.int32), batched)
        new_alive = self.alive_mask.at[entity_id].set(True)
        new_stores = dict(self.component_stores)
        new_stores[name] = store
        return replace(self, component_stores=new_stores, alive_mask=new_alive)

    def remove_component_from_entity(
        self, name: ComponentName, entity_id: EntityId
    ) -> "WorldState":
        store = self._get_store(name)
        store = store.clear(jnp.asarray([entity_id], dtype=jnp.int32))
        new_stores = dict(self.component_stores)
        new_stores[name] = store
        return replace(self, component_stores=new_stores)

    def register_event_buffer(self, specification: EventSpecification) -> "WorldState":
        if specification.name in self.event_buffers:
            raise ValueError(f"Event already registered: {specification.name}")
        buf = EventBuffer.from_specification(specification)
        new_bufs = dict(self.event_buffers)
        new_bufs[specification.name] = buf
        return replace(self, event_buffers=new_bufs)

    def reset_event_buffers(self) -> "WorldState":
        new_bufs = {k: v.clear() for k, v in self.event_buffers.items()}
        return replace(self, event_buffers=new_bufs)

    def write_event_buffer(
        self, name: str, payloads: jnp.ndarray, count: jnp.ndarray
    ) -> "WorldState":
        buf = self.event_buffers[name].overwrite(payloads, count)
        new_bufs = dict(self.event_buffers)
        new_bufs[name] = buf
        return replace(self, event_buffers=new_bufs)

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
            self.event_buffers,
        )
        aux = self.capacity
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        component_stores, alive_mask, random_key, time_step, event_buffers = children
        return cls(
            component_stores, alive_mask, random_key, time_step, aux, event_buffers
        )

    def _get_store(self, name: ComponentName) -> ComponentStore:
        try:
            return self.component_stores[name]
        except KeyError as e:
            raise KeyError(f"Component not registered: {name}") from e

    def _with_store(self, name: ComponentName, store: ComponentStore) -> "WorldState":
        new_stores = dict(self.component_stores)
        new_stores[name] = store
        return replace(self, component_stores=new_stores)


class World:

    def __init__(self, capacity: int, key: Optional[Key] = None):
        self._capacity = int(capacity)
        self._registry = EntityRegistry.create(self._capacity)
        self._world = WorldState.create(
            capacity=self._capacity,
            key=(key if key is not None else jax.random.PRNGKey(0)),
        )
        self._world = self._world.with_alive_mask(self._registry.alive_mask)
        self._systems: tuple[System, ...] = tuple()

    def register_component(self, specification: ComponentSpecification) -> "World":
        self._world = self._world.register_component(specification)
        return self

    def attach_component(
        self, entity_id: EntityId, name: ComponentName, value: Any
    ) -> "World":
        if not bool(self._world.alive_mask[entity_id]):
            raise RuntimeError(f"Entity {entity_id} is not alive; spawn first.")
        self._world = self._world.add_component_to_entity(name, entity_id, value)
        return self

    def detach_component(self, entity_id: EntityId, name: ComponentName) -> "World":
        self._world = self._world.remove_component_from_entity(name, entity_id)
        return self

    def register_event_buffer(self, specification: EventSpecification) -> "World":
        self._world = self._world.register_event_buffer(specification)
        return self

    def add_systems(self, *systems: System) -> "World":
        self._systems = tuple([*self._systems, *systems])
        return self

    def spawn(self, **components: Any) -> EntityId:
        eid = self._registry.spawn()
        self._world = self._world.with_alive_mask(self._registry.alive_mask)
        for cname, value in components.items():
            self.attach_component(eid, cname, value)
        return eid

    def despawn(self, entity_id: EntityId, clear_components: bool = True) -> "World":
        if clear_components:
            # Clear all component rows for this entity
            for name, store in self._world.component_stores.items():
                if bool(store.alive_mask[entity_id]):
                    self._world = self._world.remove_component_from_entity(
                        name, entity_id
                    )
        self._registry.despawn(entity_id)
        self._world = self._world.with_alive_mask(self._registry.alive_mask)
        return self

    def build_step(self) -> Callable[[WorldState, Mapping[str, Array]], WorldState]:
        return build_step(self._systems)

    def step(self, inputs: Mapping[str, Array] | None = None) -> "World":
        """Run one step over systems; updates internal WorldState."""
        if inputs is None:
            inputs = {}
        step_fn = self.build_step()
        self._world = step_fn(self._world, inputs)
        return self

    @property
    def state(self) -> WorldState:
        return self._world

    @property
    def capacity(self) -> int:
        return self._capacity

    def get_store(self, name: ComponentName) -> ComponentStore:
        return self._world.component_stores[name]
