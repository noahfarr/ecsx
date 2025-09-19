from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util as jtu

from .component_specification import ComponentSpecification
from .typing import PyTree


@jtu.register_pytree_node_class
@dataclass
class ComponentStore:
    """
    ECS Component storage for dataclass/PyTree components.

    - spec:       ComponentSpecification (holds name + single-entity prototype)
    - data:       PyTree of arrays; each leaf has shape (capacity, *leaf_shape_of_component)
    - alive_mask: (capacity,) True where entity HAS this component
    """

    spec: ComponentSpecification
    data: PyTree
    alive_mask: jnp.ndarray

    @staticmethod
    def from_spec(spec: ComponentSpecification, capacity: int) -> "ComponentStore":
        """
        Build storage with every slot initialized to the component's default (prototype) and alive=False.
        """

        def _make_leaf(default_leaf):
            leaf = jnp.asarray(default_leaf)
            return jnp.broadcast_to(leaf, (capacity, *leaf.shape))

        data = jtu.tree_map(_make_leaf, spec.prototype)
        alive_mask = jnp.zeros((capacity,), dtype=bool)
        return ComponentStore(spec, data, alive_mask)

    @property
    def capacity(self) -> int:
        first_leaf = jtu.tree_leaves(self.data)[0]
        return int(first_leaf.shape[0])

    def read(self, indices: jnp.ndarray) -> PyTree:
        idx = jnp.asarray(indices, dtype=jnp.int32).reshape(-1)
        return jtu.tree_map(lambda arr: arr[idx], self.data)

    def write(self, indices: jnp.ndarray, value: PyTree) -> "ComponentStore":
        idx = jnp.asarray(indices, dtype=jnp.int32).reshape(-1)
        n = int(idx.shape[0])

        if jtu.tree_structure(value) != self.spec.treedef():
            raise ValueError(
                f"{self.spec.name}: write value structure does not match component prototype."
            )

        def _setter(arr, v_leaf, proto_leaf):
            v = jnp.asarray(v_leaf, dtype=jnp.asarray(proto_leaf).dtype)
            expected_shape = (n, *jnp.asarray(proto_leaf).shape)
            if v.shape != expected_shape:
                raise ValueError(
                    f"{self.spec.name}: write leaf shape {v.shape} != {expected_shape}"
                )
            return arr.at[idx].set(v)

        new_data = jtu.tree_map(_setter, self.data, value, self.spec.prototype)
        new_mask = self.alive_mask.at[idx].set(True)
        return ComponentStore(self.spec, new_data, new_mask)

    def clear(self, indices: jnp.ndarray) -> "ComponentStore":
        idx = jnp.asarray(indices, dtype=jnp.int32).reshape(-1)

        def _default_row(prototype_leaf):
            leaf = jnp.asarray(prototype_leaf)
            return jnp.broadcast_to(leaf, (idx.shape[0], *leaf.shape))

        default_batch = jtu.tree_map(_default_row, self.spec.prototype)

        def _clear_leaf(arr, v_leaf):
            return arr.at[idx].set(v_leaf)

        new_data = jtu.tree_map(_clear_leaf, self.data, default_batch)
        new_mask = self.alive_mask.at[idx].set(False)
        return ComponentStore(self.spec, new_data, new_mask)

    def tree_flatten(self):
        children = (self.data, self.alive_mask)
        aux = self.spec
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        data, alive_mask = children
        return cls(aux, data, alive_mask)
