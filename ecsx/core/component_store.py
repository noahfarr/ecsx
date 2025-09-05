from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util

from .component_specification import ComponentSpecification


@tree_util.register_pytree_node_class
@dataclass
class ComponentStore:
    """
    Columnar storage for a single component type.

    Arrays are functional (no in-place Python mutation).
    - data: (capacity, *spec.shape)
    - alive_mask: (capacity,) True where entity HAS this component
    """

    spec: ComponentSpecification
    data: jnp.ndarray
    alive_mask: jnp.ndarray  # bool (capacity,)

    @staticmethod
    def from_spec(spec: ComponentSpecification, capacity: int) -> "ComponentStore":
        # Default row matches the spec's shape/dtype
        default_row = jnp.asarray(spec.default_value, spec.dtype)
        data = jnp.broadcast_to(default_row, (capacity, *spec.shape))
        # IMPORTANT: no entity has this component at start
        alive_mask = jnp.zeros((capacity,), dtype=bool)
        return ComponentStore(spec, data, alive_mask)

    @property
    def capacity(self) -> int:
        return int(self.data.shape[0])

    def read(self, indices: jnp.ndarray) -> jnp.ndarray:
        indices = jnp.asarray(indices, dtype=jnp.int32).reshape(-1)
        return self.data[indices]

    def write(self, indices: jnp.ndarray, value: jnp.ndarray) -> "ComponentStore":
        # Expect value shape == (len(indices), *spec.shape)
        indices = jnp.asarray(indices, dtype=jnp.int32).reshape(-1)
        value = jnp.asarray(value, self.spec.dtype)
        expected = (int(indices.shape[0]), *self.spec.shape)
        if value.shape != expected:
            raise ValueError(
                f"{self.spec.name}: write shape {value.shape} != {expected}"
            )
        new_data = self.data.at[indices].set(value)
        new_mask = self.alive_mask.at[indices].set(True)
        return ComponentStore(self.spec, new_data, new_mask)

    def clear(self, indices: jnp.ndarray) -> "ComponentStore":
        indices = jnp.asarray(indices, dtype=jnp.int32).reshape(-1)
        default_row = jnp.asarray(self.spec.default_value, self.spec.dtype)
        new_data = self.data.at[indices].set(default_row)
        new_mask = self.alive_mask.at[indices].set(False)
        return ComponentStore(self.spec, new_data, new_mask)

    # ---- PyTree integration
    def tree_flatten(self):
        children = (self.data, self.alive_mask)
        aux = self.spec
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        data, alive_mask = children
        return cls(aux, data, alive_mask)
