import jax.numpy as jnp
from ecsx.core.event_specification import EventSpecification


def test_event_specification_fields():
    spec = EventSpecification("Evt", (3,), jnp.int32, capacity=5)
    assert spec.name == "Evt"
    assert spec.shape == (3,)
    assert spec.dtype == jnp.int32
    assert spec.capacity == 5
