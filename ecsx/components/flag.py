from dataclasses import dataclass
from jax import tree_util as jtu

from ecsx.components import Position, Carriable, Agent


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class Flag:
    position: Position
    owner: Agent
    carriable: Carriable

    def tree_flatten(self):
        return (self.position, self.owner, self.carriable), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
