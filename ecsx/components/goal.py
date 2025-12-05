from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util as jtu

from ecsx.components import Position, Team, Carriable


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class Goal:
    position: Position

    def tree_flatten(self):
        return (self.position,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
