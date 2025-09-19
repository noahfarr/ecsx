import jax.numpy as jnp

from ecsx.core.event_specification import EventSpecification


def test_event_specification_fields():
    spec = EventSpecification(
        name="TestEvent",
        shape=(2, 3),
        dtype=jnp.float32,
        capacity=5,
    )

    assert spec.name == "TestEvent"
    assert spec.shape == (2, 3)
    assert spec.dtype == jnp.float32
    assert spec.capacity == 5
