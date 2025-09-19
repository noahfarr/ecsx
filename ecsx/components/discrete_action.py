from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util as jtu


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class DiscreteAction:
    value: jnp.ndarray
    n: jnp.ndarray
    dtype: jnp.dtype = jnp.int32

    def tree_flatten(self):
        return (self.value, self.n, self.dtype), {"dtype": self.dtype}

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, dtype=aux["dtype"])
