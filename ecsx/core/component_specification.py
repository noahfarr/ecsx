from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util as jtu

from .typing import PyTree


@dataclass(frozen=True)
class ComponentSpecification:
    """
    Specification for a dataclass (PyTree) component.

    - name:     logical component name
    - prototype: a SINGLE-ENTITY default instance of the component (PyTree).
                 Each leaf should be a jnp.ndarray (or array-like) with shape leaf_shape (no batch dim).
    """

    name: str
    prototype: PyTree

    def treedef(self) -> jtu.PyTreeDef:
        return jtu.tree_structure(self.prototype)

    def leaves(self):
        return jtu.tree_leaves(self.prototype)

    def leaf_shapes(self) -> list[tuple[int, ...]]:
        return [tuple(jnp.asarray(leave).shape) for leave in self.leaves()]

    def leaf_dtypes(self) -> list[jnp.dtype]:
        return [jnp.asarray(leave).dtype for leave in self.leaves()]
