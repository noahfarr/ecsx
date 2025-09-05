from dataclasses import replace
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp

from ..core.typing import Array, Key, ComponentName, EntityId
from ..core.entity_registry import EntityRegistry
from ..core.component_specification import ComponentSpecification
from ..core.component_store import ComponentStore
from ..core.world_state import WorldState
from ..events.event_specification import EventSpecification
from ..systems.pipeline_builder import build_step


SystemFn = Callable[[WorldState, Key, Mapping[str, Array]], WorldState]


class World:
    """
    Host-side facade managing:
      - Entity ids (EntityRegistry)
      - Component registration and attach/detach
      - System list and compiled step
      - Event buffer registration
      - Serialization hooks (via serialization.py)
    """

    def __init__(self, capacity: int, key: Optional[Key] = None):
        self._capacity = int(capacity)
        self._registry = EntityRegistry.create(self._capacity)
        self._world = WorldState.create(
            capacity=self._capacity,
            key=(key if key is not None else jax.random.PRNGKey(0)),
        )
        # sync alive mask initially
        self._world = self._world.with_alive_mask(self._registry.alive_mask)
        self._systems: Tuple[SystemFn, ...] = tuple()
        self._compiled_step: Optional[Callable[[WorldState, Mapping[str, Array]], WorldState]] = None

    def register_component(self, specification: ComponentSpecification) -> "World":
        self._world = self._world.register_component(specification)
        return self

    def register_event_buffer(self, specification: EventSpecification) -> "World":
        self._world = self._world.register_event_buffer(specification)
        return self

    def add_systems(self, *systems: SystemFn) -> "World":
        self._systems = tuple([*self._systems, *systems])
        self._compiled_step = None
        return self

    def attach_component(self, entity_id: EntityId, name: ComponentName, value: Array) -> "World":
        # require alive
        if not bool(self._world.alive_mask[entity_id]):
            raise RuntimeError(f"Entity {entity_id} is not alive; spawn first.")
        self._world = self._world.add_component_to_entity(name, entity_id, value)
        return self

    def detach_component(self, entity_id: EntityId, name: ComponentName) -> "World":
        self._world = self._world.remove_component_from_entity(name, entity_id)
        return self

    def spawn(self, **components: Array) -> EntityId:
        """Spawn an entity and optionally attach components by name=value."""
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
                    self._world = self._world.remove_component_from_entity(name, entity_id)
        self._registry.despawn(entity_id)
        self._world = self._world.with_alive_mask(self._registry.alive_mask)
        return self

    def build_step(self) -> Callable[[WorldState, Mapping[str, Array]], WorldState]:
        if self._compiled_step is None:
            self._compiled_step = build_step(self._systems)
        return self._compiled_step

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

