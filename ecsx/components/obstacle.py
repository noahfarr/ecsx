from dataclasses import dataclass
from jax import tree_util as jtu

from ecsx.components import Position


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class Obstacle:
    position: Position

    def tree_flatten(self):
        return (self.position,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
