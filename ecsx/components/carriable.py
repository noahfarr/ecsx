from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util as jtu


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class Carriable:
    eid: jnp.ndarray
    carried: jnp.ndarray

    def tree_flatten(self):
        return (self.eid, self.carried), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
