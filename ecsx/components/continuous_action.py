from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util as jtu


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class ContinuousAction:
    value: jnp.ndarray
    shape: jnp.ndarray
    low: jnp.ndarray
    high: jnp.ndarray
    dtype: jnp.dtype

    def tree_flatten(self):
        return (self.value, self.shape, self.low, self.high, self.dtype), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
