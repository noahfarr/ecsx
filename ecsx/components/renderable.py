from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util as jtu


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class Renderable:
    id: jnp.ndarray

    def tree_flatten(self):
        return (self.id,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
