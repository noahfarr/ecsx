from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util

from .event_specification import EventSpecification


@tree_util.register_pytree_node_class
@dataclass
class EventBuffer:
    specification: EventSpecification
    data: jnp.ndarray  # (capacity, *shape)
    count: jnp.ndarray  # scalar int32

    @staticmethod
    def from_specification(spec: EventSpecification) -> "EventBuffer":
        data = jnp.zeros((spec.capacity, *spec.shape), dtype=spec.dtype)
        return EventBuffer(spec, data, jnp.array(0, dtype=jnp.int32))

    def clear(self) -> "EventBuffer":
        return EventBuffer(self.specification, self.data, jnp.array(0, dtype=jnp.int32))

    def overwrite(self, payloads: jnp.ndarray, count: jnp.ndarray) -> "EventBuffer":
        assert payloads.shape == self.data.shape, (payloads.shape, self.data.shape)
        return EventBuffer(self.specification, payloads, jnp.asarray(count, jnp.int32))

    def tree_flatten(self):
        return (self.data, self.count), self.specification

    @classmethod
    def tree_unflatten(cls, spec, children):
        data, count = children
        return cls(spec, data, count)
