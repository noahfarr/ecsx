from dataclasses import dataclass, replace

import jax.numpy as jnp
from jax import tree_util

from ecsx.core.typing import Array, Key, ComponentName, EntityId
from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.component_table import ComponentTable


@tree_util.register_pytree_node_class
@dataclass
class WorldState:
    """
    JAX-friendly immutable world state.
    - component_tables: mapping[name] -> ComponentTable
    - alive_mask: (capacity,)
    - random_key: PRNG key
    - time_step: int scalar array
    """

    component_tables: dict[ComponentName, ComponentTable]
    alive_mask: Array
    random_key: Key
    time_step: Array  # int32
    capacity: int

    @staticmethod
    def create(capacity: int, key: Key) -> "WorldState":
        return WorldState(
            component_tables={},
            alive_mask=jnp.zeros((capacity,), dtype=bool),
            random_key=key,
            time_step=jnp.array(0, dtype=jnp.int32),
            capacity=capacity,
        )

    def register_component(self, spec: ComponentSpecification) -> "WorldState":
        if spec.name in self.component_tables:
            raise ValueError(f"Component already registered: {spec.name}")
        table = ComponentTable.from_spec(spec, self.capacity)
        new_tables = dict(self.component_tables)
        new_tables[spec.name] = table
        return replace(self, component_tables=new_tables)

    def add_component_to_entity(
        self, name: ComponentName, entity_id: EntityId, value: Array
    ) -> "WorldState":
        table = self._get_table(name)
        value = table.spec.validate_value(value)
        table = table.write(
            jnp.asarray([entity_id], dtype=jnp.int32), value[jnp.newaxis, ...]
        )
        # ensure entity is considered alive if it now has any component
        new_alive = self.alive_mask.at[entity_id].set(True)
        new_tables = dict(self.component_tables)
        new_tables[name] = table
        return replace(self, component_tables=new_tables, alive_mask=new_alive)

    def remove_component_from_entity(
        self, name: ComponentName, entity_id: EntityId
    ) -> "WorldState":
        table = self._get_table(name)
        table = table.clear(jnp.asarray([entity_id], dtype=jnp.int32))
        # Do NOT auto-clear alive here; liveness is controlled by the registry / host sync.
        new_tables = dict(self.component_tables)
        new_tables[name] = table
        return replace(self, component_tables=new_tables)

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
            self.component_tables,
            self.alive_mask,
            self.random_key,
            self.time_step,
        )
        aux = self.capacity
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        component_tables, alive_mask, random_key, time_step = children
        return cls(component_tables, alive_mask, random_key, time_step, aux)

    def _get_table(self, name: ComponentName) -> ComponentTable:
        try:
            return self.component_tables[name]
        except KeyError as e:
            raise KeyError(f"Component not registered: {name}") from e

    def _with_table(self, name: ComponentName, table: ComponentTable) -> "WorldState":
        new_tables = dict(self.component_tables)
        new_tables[name] = table
        return replace(self, component_tables=new_tables)
