from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util as jtu


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class Team:
    id: jnp.ndarray

    def tree_flatten(self):
        return (self.id,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        id = children
        return cls(id)
