from dataclasses import dataclass
from jax import tree_util as jtu
import jax.numpy as jnp

from ecsx.components import Position


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class Direction:
    value: jnp.ndarray

    def tree_flatten(self):
        return (self.value,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
